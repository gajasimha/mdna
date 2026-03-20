import argparse
import threading
import time

import dash
from dash import Dash, dcc, html, Input, Output, State
import mdtraj as md
import numpy as np
import plotly.graph_objects as go
from mdna.geometry import ReferenceBase


class OnDemandBaseFramesViewer:
    def __init__(self, traj, verbose=True):
        self.traj = traj
        self.verbose = verbose

        self.residue_refs = self._build_reference_bases()
        if not self.residue_refs:
            raise ValueError("No residues could be converted to ReferenceBase.")

        self.all_atom_idx = self._build_all_atom_indices()

        # Precompute arrays once
        self.res_labels = [f"{item['residue'].index}:{item['residue'].name}" for item in self.residue_refs]

        self.origins = np.array([item["ref"].b_R for item in self.residue_refs], dtype=float)
        self.bL = np.array([item["ref"].b_L for item in self.residue_refs], dtype=float)
        self.bD = np.array([item["ref"].b_D for item in self.residue_refs], dtype=float)
        self.bN = np.array([item["ref"].b_N for item in self.residue_refs], dtype=float)

        # (n_res, n_frames, 3) -> (n_frames, n_res, 3)
        self.origins = np.swapaxes(self.origins, 0, 1)
        self.bL = np.swapaxes(self.bL, 0, 1)
        self.bD = np.swapaxes(self.bD, 0, 1)
        self.bN = np.swapaxes(self.bN, 0, 1)

        # Cache atoms once
        self.atom_coords = self.traj.xyz[:, self.all_atom_idx, :]

    def _build_reference_bases(self):
        refs = []
        for res in self.traj.topology.residues:
            atom_indices = [a.index for a in res.atoms]
            if not atom_indices:
                continue

            subtraj = self.traj.atom_slice(atom_indices)
            try:
                ref = ReferenceBase(subtraj)
                refs.append(
                    {
                        "residue": res,
                        "ref": ref,
                        "atom_indices": atom_indices,
                    }
                )
                if self.verbose:
                    print(f"OK resid {res.index:>3} {res.name:<4} base_type={ref.base_type}")
            except Exception as e:
                if self.verbose:
                    print(f"SKIP resid {res.index:>3} {res.name:<4} reason={e}")
        return refs

    def _build_all_atom_indices(self):
        return np.array(
            sorted(set(idx for item in self.residue_refs for idx in item["atom_indices"])),
            dtype=int,
        )

    def _segment_arrays(self, origins, vecs, length):
        ends = origins + length * vecs
        n = origins.shape[0]

        x = np.empty(3 * n, dtype=object)
        y = np.empty(3 * n, dtype=object)
        z = np.empty(3 * n, dtype=object)

        x[0::3] = origins[:, 0]
        x[1::3] = ends[:, 0]
        x[2::3] = None

        y[0::3] = origins[:, 1]
        y[1::3] = ends[:, 1]
        y[2::3] = None

        z[0::3] = origins[:, 2]
        z[1::3] = ends[:, 2]
        z[2::3] = None

        return x.tolist(), y.tolist(), z.tolist(), ends

    def _arrowhead_arrays(self, ends, vecs, length, head_frac=0.30, head_angle_deg=28):
        n = ends.shape[0]
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        head_length = length * head_frac
        theta = np.deg2rad(head_angle_deg)

        helpers = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        mask = np.abs(np.sum(helpers * vecs, axis=1)) > 0.9
        helpers[mask] = np.array([0.0, 1.0, 0.0])

        u = np.cross(vecs, helpers)
        u /= np.linalg.norm(u, axis=1, keepdims=True)

        h1 = -np.cos(theta) * vecs + np.sin(theta) * u
        h2 = -np.cos(theta) * vecs - np.sin(theta) * u

        p1 = ends + head_length * h1
        p2 = ends + head_length * h2

        # two short segments per arrowhead side, separated by None
        x = np.empty(6 * n, dtype=object)
        y = np.empty(6 * n, dtype=object)
        z = np.empty(6 * n, dtype=object)

        x[0::6] = ends[:, 0]
        x[1::6] = p1[:, 0]
        x[2::6] = None
        x[3::6] = ends[:, 0]
        x[4::6] = p2[:, 0]
        x[5::6] = None

        y[0::6] = ends[:, 1]
        y[1::6] = p1[:, 1]
        y[2::6] = None
        y[3::6] = ends[:, 1]
        y[4::6] = p2[:, 1]
        y[5::6] = None

        z[0::6] = ends[:, 2]
        z[1::6] = p1[:, 2]
        z[2::6] = None
        z[3::6] = ends[:, 2]
        z[4::6] = p2[:, 2]
        z[5::6] = None

        return x.tolist(), y.tolist(), z.tolist()

    def _axis_traces(self, origins, vecs, length, color, name, line_width=6):
        xs, ys, zs, ends = self._segment_arrays(origins, vecs, length)
        hx, hy, hz = self._arrowhead_arrays(ends, vecs, length)

        shaft = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )

        heads = go.Scatter3d(
            x=hx,
            y=hy,
            z=hz,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )

        return [shaft, heads]

    def _get_global_bounds_equal(
        self,
        length=0.25,
        show_atoms=False,
        show_L=False,
        show_D=False,
        show_N=True,
        pad=0.10,
    ):
        pts = [self.origins.reshape(-1, 3)]

        if show_atoms:
            pts.append(self.atom_coords.reshape(-1, 3))
        if show_L:
            pts.append((self.origins + length * self.bL).reshape(-1, 3))
        if show_D:
            pts.append((self.origins + length * self.bD).reshape(-1, 3))
        if show_N:
            pts.append((self.origins + length * self.bN).reshape(-1, 3))

        pts = np.vstack(pts)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins)
        radius *= (1.0 + pad)

        return center - radius, center + radius

    def make_figure(
        self,
        frame_idx,
        length=0.25,
        show_atoms=True,
        show_labels=False,
        show_L=False,
        show_D=False,
        show_N=True,
    ):
        traces = []

        origins = self.origins[frame_idx]

        if show_atoms:
            coords = self.atom_coords[frame_idx]
            traces.append(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(size=3, opacity=0.35),
                    name="atoms",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if show_L:
            traces.extend(self._axis_traces(origins, self.bL[frame_idx], length, "red", "L"))

        if show_D:
            traces.extend(self._axis_traces(origins, self.bD[frame_idx], length, "green", "D"))

        if show_N:
            traces.extend(self._axis_traces(origins, self.bN[frame_idx], length, "blue", "N"))

        if show_labels:
            traces.append(
                go.Scatter3d(
                    x=origins[:, 0],
                    y=origins[:, 1],
                    z=origins[:, 2],
                    mode="text",
                    text=self.res_labels,
                    textposition="top center",
                    name="labels",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        mins, maxs = self._get_global_bounds_equal(
            length=length,
            show_atoms=show_atoms,
            show_L=show_L,
            show_D=show_D,
            show_N=show_N,
            pad=0.10,
        )
        xmin, ymin, zmin = mins
        xmax, ymax, zmax = maxs

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Base reference frames — frame {frame_idx}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
                xaxis=dict(range=[xmin, xmax], autorange=False),
                yaxis=dict(range=[ymin, ymax], autorange=False),
                zaxis=dict(range=[zmin, zmax], autorange=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig


def build_app(viewer, length, show_atoms, show_labels, show_L, show_D, show_N, fps):
    app = Dash(__name__)

    interval_ms = int(1000 / fps)

    app.layout = html.Div(
        [
            dcc.Graph(
                id="graph",
                figure=viewer.make_figure(
                    frame_idx=0,
                    length=length,
                    show_atoms=show_atoms,
                    show_labels=show_labels,
                    show_L=show_L,
                    show_D=show_D,
                    show_N=show_N,
                ),
                style={"height": "90vh"},
            ),
            html.Div(
                [
                    html.Button("Play / Pause", id="play-btn", n_clicks=0),
                    dcc.Slider(
                        id="frame-slider",
                        min=0,
                        max=viewer.traj.n_frames - 1,
                        step=1,
                        value=0,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    dcc.Interval(id="timer", interval=interval_ms, n_intervals=0, disabled=True),
                    dcc.Store(id="playing", data=False),
                ],
                style={"padding": "10px"},
            ),
        ]
    )

    @app.callback(
        Output("playing", "data"),
        Output("timer", "disabled"),
        Input("play-btn", "n_clicks"),
        State("playing", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(_, playing):
        playing = not playing
        return playing, (not playing)

    @app.callback(
        Output("frame-slider", "value"),
        Input("timer", "n_intervals"),
        State("frame-slider", "value"),
        State("playing", "data"),
        prevent_initial_call=True,
    )
    def advance_frame(_, current_frame, playing):
        if not playing:
            return current_frame
        return (current_frame + 1) % viewer.traj.n_frames

    @app.callback(
        Output("graph", "figure"),
        Input("frame-slider", "value"),
    )
    def update_graph(frame_idx):
        return viewer.make_figure(
            frame_idx=frame_idx,
            length=length,
            show_atoms=show_atoms,
            show_labels=show_labels,
            show_L=show_L,
            show_D=show_D,
            show_N=show_N,
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="On-demand base-frame viewer with arrows.")
    parser.add_argument("--traj", required=True, help="Trajectory file, e.g. traj.xtc")
    parser.add_argument("--top", required=True, help="Topology file, e.g. top.pdb or conf.gro")
    parser.add_argument("--stride", type=int, default=20, help="Take every nth frame")
    parser.add_argument("--length", type=float, default=0.25, help="Axis length")
    parser.add_argument("--fps", type=float, default=8.0, help="Playback frames per second")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    return parser.parse_args()


def main():
    args = parse_args()

    traj = md.load(args.traj, top=args.top)
    if args.stride > 1:
        traj = traj[::args.stride]

    viewer = OnDemandBaseFramesViewer(traj, verbose=False)

    app = build_app(
        viewer=viewer,
        length=args.length,
        show_atoms=True,
        show_labels=True,
        show_L=False,
        show_D=False,
        show_N=True,
        fps=args.fps,
    )

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()