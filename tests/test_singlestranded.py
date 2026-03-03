import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "mdna"

sys.modules.setdefault("mdtraj", types.ModuleType("mdtraj"))

if "mdna" not in sys.modules:
    package = types.ModuleType("mdna")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["mdna"] = package


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_module("mdna.utils", PACKAGE_ROOT / "utils.py")
geometry = load_module("mdna.geometry", PACKAGE_ROOT / "geometry.py")
NucleicFrames = geometry.NucleicFrames
SingleStrandFrames = geometry.SingleStrandFrames


class DummyAtom:
    def __init__(self, index):
        self.index = index


class DummyResidue:
    def __init__(self, atom_indices):
        self.atoms = [DummyAtom(index) for index in atom_indices]


class DummyTrajectory:
    def __init__(self, topology=None, n_frames=2):
        self.topology = topology
        self.n_frames = n_frames
        self.slice_calls = []

    def atom_slice(self, indices):
        indices = list(indices)
        self.slice_calls.append(indices)
        return f"slice-{indices}"


def make_uninitialized_strand(**attrs):
    strand = SingleStrandFrames.__new__(SingleStrandFrames)
    for name, value in attrs.items():
        setattr(strand, name, value)
    return strand


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("get_residues", (2,), {"reverse": True}),
        ("load_reference_bases", (), {}),
        ("_prepare_reference_fit_data", (), {}),
        ("_get_fitted_base_vectors", ("res", "ref", "default"), {}),
        ("get_base_vectors", ("res",), {}),
        ("reshape_input", ("A", "B"), {"is_step": True}),
        ("compute_parameters", ("rot_a", "rot_b", "origin_a", "origin_b"), {}),
        ("calculate_parameters", ("frames_a", "frames_b"), {"is_step": True}),
    ],
)
def test_delegated_methods_forward_to_nucleicframes(monkeypatch, method_name, args, kwargs):
    strand = make_uninitialized_strand()
    sentinel = object()

    def fake(*call_args, **call_kwargs):
        assert call_args[0] is strand
        assert call_args[1:] == args
        assert call_kwargs == kwargs
        return sentinel

    monkeypatch.setattr(NucleicFrames, method_name, fake)

    result = getattr(SingleStrandFrames, method_name)(strand, *args, **kwargs)

    assert result is sentinel


def test_init_sets_attributes_and_runs_analysis(monkeypatch):
    residues = ["res-1", "res-2"]
    base_frames = {"res-1": "frame-1"}
    reference_fit_data = {"A": "canonical-A"}
    call_order = []

    def fake_prepare(self):
        call_order.append("prepare")
        return reference_fit_data

    def fake_get_residues(self, chain_index, reverse=False):
        call_order.append(("residues", chain_index, reverse))
        return residues

    def fake_get_base_reference_frames(self):
        call_order.append("frames")
        return base_frames

    def fake_analyse_frames(self):
        call_order.append("analyse")

    monkeypatch.setattr(SingleStrandFrames, "_prepare_reference_fit_data", fake_prepare)
    monkeypatch.setattr(SingleStrandFrames, "get_residues", fake_get_residues)
    monkeypatch.setattr(SingleStrandFrames, "get_base_reference_frames", fake_get_base_reference_frames)
    monkeypatch.setattr(SingleStrandFrames, "analyse_frames", fake_analyse_frames)

    traj = DummyTrajectory(topology=SimpleNamespace(name="topology"))
    strand = SingleStrandFrames(traj, chainid=3, fit_reference=True)

    assert strand.traj is traj
    assert strand.top is traj.topology
    assert strand.chainid == 3
    assert strand.fit_reference is True
    assert strand.reference_base_map == {"U": "T"}
    assert strand.reference_fit_data is reference_fit_data
    assert strand.residues is residues
    assert strand.base_frames is base_frames
    assert call_order == ["prepare", ("residues", 3, False), "frames", "analyse"]


def test_init_skips_reference_preparation_when_fitting_disabled(monkeypatch):
    def fail_prepare(self):
        raise AssertionError("_prepare_reference_fit_data should not be called")

    monkeypatch.setattr(SingleStrandFrames, "_prepare_reference_fit_data", fail_prepare)
    monkeypatch.setattr(SingleStrandFrames, "get_residues", lambda self, chain_index, reverse=False: [])
    monkeypatch.setattr(SingleStrandFrames, "get_base_reference_frames", lambda self: {})
    monkeypatch.setattr(SingleStrandFrames, "analyse_frames", lambda self: None)

    traj = DummyTrajectory(topology=SimpleNamespace(name="topology"))
    strand = SingleStrandFrames(traj, fit_reference=False)

    assert strand.reference_fit_data == {}


def test_get_base_reference_frames_slices_each_residue_and_collects_vectors():
    residues = [DummyResidue([1, 2]), DummyResidue([7, 9, 11])]
    traj = DummyTrajectory()
    strand = make_uninitialized_strand(traj=traj, residues=residues)

    def fake_get_base_vectors(res_traj):
        return f"vectors-for-{res_traj}"

    strand.get_base_vectors = fake_get_base_vectors

    reference_frames = SingleStrandFrames.get_base_reference_frames(strand)

    assert traj.slice_calls == [[1, 2], [7, 9, 11]]
    assert reference_frames[residues[0]] == "vectors-for-slice-[1, 2]"
    assert reference_frames[residues[1]] == "vectors-for-slice-[7, 9, 11]"


def test_analyse_frames_builds_step_parameters_for_multiple_residues():
    residues = [DummyResidue([0]), DummyResidue([1]), DummyResidue([2])]
    base_frames = {
        residue: np.full((2, 4, 3), fill_value=index + 1, dtype=float)
        for index, residue in enumerate(residues)
    }
    expected_step_params = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    strand = make_uninitialized_strand(
        base_frames=base_frames,
        residues=residues,
        traj=SimpleNamespace(n_frames=2),
    )

    def fake_calculate_parameters(frames_a, frames_b, is_step=False):
        assert is_step is True
        assert frames_a.shape == (2, 2, 4, 3)
        assert frames_b.shape == (2, 2, 4, 3)
        return expected_step_params, "unused"

    strand.calculate_parameters = fake_calculate_parameters

    SingleStrandFrames.analyse_frames(strand)

    assert strand.frames.shape == (3, 2, 4, 3)
    assert np.array_equal(strand.frames[0], base_frames[residues[0]])
    assert np.array_equal(strand.frames[2], base_frames[residues[2]])
    assert strand.step_parameter_names == ["shift", "slide", "rise", "tilt", "roll", "twist"]
    assert strand.base_parameter_names == ["shear", "stretch", "stagger", "buckle", "propeller", "opening"]
    assert np.array_equal(strand.step_params, expected_step_params)
    assert strand.names == strand.step_parameter_names
    assert strand.parameters is strand.step_params


def test_analyse_frames_uses_zero_step_parameters_for_single_residue():
    residue = DummyResidue([0, 1])
    strand = make_uninitialized_strand(
        base_frames={residue: np.ones((3, 4, 3), dtype=float)},
        residues=[residue],
        traj=SimpleNamespace(n_frames=3),
    )

    def fail_calculate_parameters(*args, **kwargs):
        raise AssertionError("calculate_parameters should not be called for a single residue")

    strand.calculate_parameters = fail_calculate_parameters

    SingleStrandFrames.analyse_frames(strand)

    assert strand.step_params.shape == (3, 1, 6)
    assert np.array_equal(strand.step_params, np.zeros((3, 1, 6)))


def test_get_parameters_returns_step_parameters_and_names():
    step_params = np.arange(12, dtype=float).reshape(1, 2, 6)
    strand = make_uninitialized_strand(
        step_params=step_params,
        step_parameter_names=["shift", "slide", "rise", "tilt", "roll", "twist"],
    )

    params, names = SingleStrandFrames.get_parameters(strand)

    assert params is step_params
    assert names == ["shift", "slide", "rise", "tilt", "roll", "twist"]


def test_get_parameters_rejects_base_pair_output():
    strand = make_uninitialized_strand(step_params=np.zeros((1, 1, 6)), step_parameter_names=["shift"])

    with pytest.raises(NotImplementedError, match="Base-pair parameters require paired strands"):
        SingleStrandFrames.get_parameters(strand, base=True)


def test_get_parameter_returns_requested_step_parameter_slice():
    step_names = ["shift", "slide", "rise", "tilt", "roll", "twist"]
    step_params = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    strand = make_uninitialized_strand(
        step_params=step_params,
        step_parameter_names=step_names,
        base_parameter_names=["shear", "stretch", "stagger", "buckle", "propeller", "opening"],
    )

    twist = SingleStrandFrames.get_parameter(strand, "twist")

    assert np.array_equal(twist, step_params[:, :, 5])


def test_get_parameter_rejects_base_pair_parameter_names():
    strand = make_uninitialized_strand(
        step_params=np.zeros((1, 1, 6)),
        step_parameter_names=["shift", "slide", "rise", "tilt", "roll", "twist"],
        base_parameter_names=["shear", "stretch", "stagger", "buckle", "propeller", "opening"],
    )

    with pytest.raises(NotImplementedError, match="Base-pair parameters require paired strands"):
        SingleStrandFrames.get_parameter(strand, "shear")


def test_get_parameter_rejects_unknown_parameter_names():
    strand = make_uninitialized_strand(
        step_params=np.zeros((1, 1, 6)),
        step_parameter_names=["shift", "slide", "rise", "tilt", "roll", "twist"],
        base_parameter_names=["shear", "stretch", "stagger", "buckle", "propeller", "opening"],
    )

    with pytest.raises(ValueError, match="Parameter foo not found"):
        SingleStrandFrames.get_parameter(strand, "foo")
