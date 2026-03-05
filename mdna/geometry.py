import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import quaternionic as qt
from .utils import RigidBody, get_data_file_path, get_sequence_letters
from numba import jit 
from scipy.spatial.transform import Rotation as R

NUCLEOBASE_DICT =  {'A': ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
                    'T': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'],
                    'G': ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
                    'C': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
                    'U': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],
                    'D': ['N1','C2','O2','N3','C4','C6','C14','C13','N5','C11','S12','C7','C8','C9','C10'],
                    'E': ['N9', 'C8', 'N7', 'C5', 'C6', 'N1', 'C2', 'N2', 'N3', 'C4'],
                    'L': ['C1','N1','S1','C2','C3','C4','C5','C6','C7', 'C8','C9','C10'],
                    'M': ['C1','C2','C3','C4','C5','C6','C20','C21','C22','C23','O37','C38'],
                    'B': ['N1', 'C2', 'N2', 'N3', 'C4', 'N5', 'C6', 'O6', 'C7', 'C8', 'N9'],
                    'S': ['N', 'C1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', 'ON1', 'ON2'],
                    'Z': ['C1', 'C2', 'C4', 'C6', 'C7', 'N2', 'N3', 'N5', 'O4'],
                    'P': ['N9', 'C8', 'N7', 'C6', 'N6', 'C5', 'N1', 'C2', 'O2', 'N3', 'C4']}

class ReferenceBase:
    """_summary_
    """
    def __init__(self, traj):
        """_summary_

        Args:
            traj (_type_): _description_
        """
        self.traj = traj
        # Determine base type (purine/pyrimidine/other)
        self.base_type = self.get_base_type()
        # Define the Tsukuba convention parameters
        self.tau_1, self.tau_2, self.d = np.radians(141.478), -np.radians(54.418), 0.4702     
        # Get coordinates of key atoms based on base type
        self.C1_coords, self.N_coords, self.C_coords = self.get_coordinates()
        # Calculate base reference point and base vectors
        self.b_R, self.b_L, self.b_D, self.b_N = self.calculate_base_frame()
        # self.basis = np.array([self.b_D.T, self.b_L.T, self.b_N])
    
    def _select_atom_by_name(self, name: str) -> np.ndarray:
        """_summary_

        Args:
            name (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Select an atom by name returns shape (n_frames, 1, [x,y,z])
        return np.squeeze(self.traj.xyz[:,[self.traj.topology.select(f'name {name}')[0]],:],axis=1)
        
    def get_base_type(self) -> str:
        """_summary_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # Extracts all non-hydrogen atoms from the trajectory topology
        atoms = {atom.name for atom in self.traj.topology.atoms if atom.element.symbol != 'H'}
    
        # Check each base in the dictionary to see if all its atoms are present in the extracted atoms set
        for base, base_atoms in NUCLEOBASE_DICT.items():
            if all(atom in atoms for atom in base_atoms):
                return base
        # If no base matches, raise an error
        raise ValueError("Cannot determine the base type from the PDB file.")
    
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_

        Returns:
            _type_: _description_
        """
        # Get the coordinates of key atoms based on the base type
        C1_coords = self._select_atom_by_name('"C1\'"')
        if self.base_type in ['C','T','U','D']:# "pyrimidine"
            N_coords = self._select_atom_by_name("N1")
            C_coords = self._select_atom_by_name("C2")
        elif self.base_type in ['A','G','E','B','P']:# "purine":
            N_coords = self._select_atom_by_name("N9")
            C_coords = self._select_atom_by_name("C4") 
        elif self.base_type in ['S','Z']: # Hachi pyrimidine analogues
            N_coords = self._select_atom_by_name("C1")
            C_coords = self._select_atom_by_name("C2")
        elif self.base_type in ['L']: # UBPs hydrophobic
            N_coords = self._select_atom_by_name("N1")
            C_coords = self._select_atom_by_name("C5")
        elif self.base_type in ['M']: # UBPs hydrophilic
            N_coords = self._select_atom_by_name("C1")
            C_coords = self._select_atom_by_name("C6")
        else:
            raise ValueError(f"Unsupported base type: {self.base_type}")
        return C1_coords, N_coords, C_coords
    
    
    def calculate_base_frame(self) -> np.ndarray:
        """_summary_

        Returns:
            _type_: _description_
        """

        # Calculate normal vector using cross product of vectors formed by key atoms
        #  The coords have the shape (n,1,3)
        b_N = np.cross((self.N_coords - self.C1_coords), (self.N_coords-self.C_coords), axis=1)
        b_N /= np.linalg.norm(b_N, axis=1, keepdims=True)  # Normalize b_N to have unit length

        # Compute displacement vector N-C1' 
        N_C1_vector = self.C1_coords - self.N_coords  # Pointing from N to C1'
        N_C1_vector /= np.linalg.norm(N_C1_vector, axis=1, keepdims=True)

        # Rotate N-C1' vector by angle tau_1 around b_N to get the direction for displacement
        R_b_R = RigidBody.get_rotation_matrix(self.tau_1 * b_N)
       
        # Displace N along this direction by a distance d to get b_R
        b_R = self.N_coords + np.einsum('ijk,ik->ij', R_b_R, N_C1_vector * self.d)
     
        # Take a unit vector in the N-C1' direction, rotate it around b_N by angle tau_2 to get b_L
        R_b_L = RigidBody.get_rotation_matrix(self.tau_2 * b_N)
        b_L = np.einsum('ijk,ik->ij', R_b_L, N_C1_vector) 

        # Calculate b_D using cross product of b_L and b_N
        b_D = np.cross(b_L, b_N, axis=1)
        
        return np.array([b_R, b_D, b_L, b_N])
        #return np.array([b_R, -b_D, -b_L, -b_N])

    def plot_baseframe(self,atoms=True, frame=True, ax=None,length=1):
        """_summary_

        Args:
            atoms (bool, optional): _description_. Defaults to True.
            frame (bool, optional): _description_. Defaults to True.
            ax (_type_, optional): _description_. Defaults to None.
            length (int, optional): _description_. Defaults to 1.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = False

        # Plot the DNA atoms
        if atoms:
            atoms_coords = self.traj.xyz[0]
            ax.scatter(atoms_coords[:,0], atoms_coords[:,1], atoms_coords[:,2], alpha=0.6)

        # Plot the reference frame vectors
        if frame:
            origin = self.b_R[0]
            ax.quiver(origin[0], origin[1], origin[2], 
                    self.b_L[0][0], self.b_L[0][1], self.b_L[0][2], 
                    color='r', length=length, normalize=True)
            ax.quiver(origin[0], origin[1], origin[2], 
                    self.b_D[0][0], self.b_D[0][1], self.b_D[0][2], 
                    color='g', length=length, normalize=True)
            ax.quiver(origin[0], origin[1], origin[2], 
                    self.b_N[0][0], self.b_N[0][1], self.b_N[0][2], 
                    color='b', length=length, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if fig: 
            # Make axes of equal length
            max_range = np.array([
                atoms_coords[:,0].max()-atoms_coords[:,0].min(), 
                atoms_coords[:,1].max()-atoms_coords[:,1].min(), 
                atoms_coords[:,2].max()-atoms_coords[:,2].min()
            ]).max() / 2.0

            mid_x = (atoms_coords[:,0].max()+atoms_coords[:,0].min()) * 0.5
            mid_y = (atoms_coords[:,1].max()+atoms_coords[:,1].min()) * 0.5
            mid_z = (atoms_coords[:,2].max()+atoms_coords[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.axis('equal')

class NucleicFrames:
    def _init_common(self, traj, fit_reference=False):
        self.traj = traj
        self.top = traj.topology
        self.fit_reference = fit_reference
        self.reference_base_map = {"U": "T"}
        self.reference_fit_data = self._prepare_reference_fit_data() if self.fit_reference else {}

    def __init__(self, traj, chainids=[0, 1], fit_reference=False):
        self._init_common(traj, fit_reference=fit_reference)

        self.chainids = chainids
        self.res_A = self.get_residues(chain_index=chainids[0], reverse=False)
        self.res_B = self.get_residues(chain_index=chainids[1], reverse=True)

        self.mean_reference_frames = np.empty((len(self.res_A), 1, 4, 3))
        self.base_frames = self.get_base_reference_frames()
        self.analyse_frames()

    def get_residues(self, chain_index, reverse=False):
        """Get residues from specified chain."""
        if chain_index >= len(self.top._chains):
            raise IndexError("Chain index out of range.")
        chain = self.top._chains[chain_index]
        residues = chain._residues
        return list(reversed(residues)) if reverse else residues

    def load_reference_bases(self):
        """Load reference bases from local files."""
        bases = ['C', 'G', 'T', 'A']
        return {base: md.load_hdf5(get_data_file_path(f'./atomic/bases/BDNA_{base}.h5')) for base in bases}

    def _prepare_reference_fit_data(self):
        """Prepare canonical base atom coordinates and frames for optional fitting."""
        reference_fit_data = {}
        for base, base_traj in self.load_reference_bases().items():
            ref_base = ReferenceBase(base_traj)
            atom_coords = {
                atom.name: base_traj.xyz[0, atom.index, :]
                for atom in base_traj.topology.atoms
                if atom.element.symbol != 'H'
            }
            reference_fit_data[base] = {
                'atom_coords': atom_coords,
                'frame': np.array([ref_base.b_R[0], ref_base.b_L[0], ref_base.b_D[0], ref_base.b_N[0]])
            }
        return reference_fit_data

    def _get_fitted_base_vectors(self, res, ref_base, default_vectors):
        """Fit residue atoms to canonical reference and transform canonical frame."""
        reference_key = self.reference_base_map.get(ref_base.base_type, ref_base.base_type)
        reference_data = self.reference_fit_data.get(reference_key)
        if reference_data is None:
            return default_vectors

        residue_atom_indices = {
            atom.name: atom.index
            for atom in res.topology.atoms
            if atom.element.symbol != 'H'
        }

        candidate_atoms = NUCLEOBASE_DICT.get(ref_base.base_type, [])
        common_atoms = [
            atom_name for atom_name in candidate_atoms
            if atom_name in residue_atom_indices and atom_name in reference_data['atom_coords']
        ]
        if len(common_atoms) < 3:
            return default_vectors

        reference_coords = np.array([reference_data['atom_coords'][atom_name] for atom_name in common_atoms])
        residue_coords = res.xyz[:, [residue_atom_indices[atom_name] for atom_name in common_atoms], :]

        reference_frame = reference_data['frame']
        reference_center = reference_coords.mean(axis=0)
        reference_centered = reference_coords - reference_center

        fitted_vectors = np.empty_like(default_vectors)
        for frame_index in range(residue_coords.shape[0]):
            frame_coords = residue_coords[frame_index]
            frame_center = frame_coords.mean(axis=0)
            frame_centered = frame_coords - frame_center
            try:
                rotation, _ = R.align_vectors(frame_centered, reference_centered)
            except ValueError:
                return default_vectors

            fitted_vectors[frame_index, 0] = rotation.apply(reference_frame[0] - reference_center) + frame_center
            fitted_vectors[frame_index, 1:] = rotation.apply(reference_frame[1:])

        return fitted_vectors

    def get_base_vectors(self, res):
        """Compute base vectors from reference base."""
        ref_base = ReferenceBase(res)
        base_vectors = np.array([ref_base.b_R, ref_base.b_L, ref_base.b_D, ref_base.b_N]).swapaxes(0,1)
        if not self.fit_reference:
            return base_vectors
        return self._get_fitted_base_vectors(res, ref_base, base_vectors)
    
    def get_base_reference_frames(self):
        """Get reference frames for each residue."""
        reference_frames = {} # Dictionary to store the base vectors for each residue
        for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            base_vectors = self.get_base_vectors(res_traj)
            reference_frames[res] = base_vectors # Store the base vectors for the residue index (with shape (n_frames, 4, 3))
        return reference_frames

    def reshape_input(self,input_A,input_B,is_step=False):
        
        """Reshape the input to the correct format for the calculations.
        
        Args:
        input_A (ndarray): Input array for the first triad.
        input_B (ndarray): Input array for the second triad.
        is_step (bool, optional): Flag indicating if the input is a single step or a trajectory. Defaults to False.
        
        Returns:
        rotation_A (ndarray): Rotation matrices of shape (n, 3, 3) for the first triad.
        rotation_B (ndarray): Rotation matrices of shape (n, 3, 3) for the second triad.
        origin_A (ndarray): Origins of shape (n, 3) for the first triad.
        origin_B (ndarray): Origins of shape (n, 3) for the second triad.
        original_shape (tuple): The original shape of the input.
        """

        # Store original shape
        original_shape = input_A.shape

        # Flatten frames to compute rotation matrices for each time step simultaneously
        input_A_ = input_A.reshape(-1,original_shape[-2],original_shape[-1])  # shape (n, 4, 3)
        input_B_ = input_B.reshape(-1,original_shape[-2],original_shape[-1])  # shape (n, 4, 3)

        # Extract the triads without origin (rotation matrices)
        rotation_A = input_A_[:,1:]  # shape (n, 3, 3)
        rotation_B = input_B_[:,1:]  # shape (n, 3, 3)

        if not is_step:
            # flip (connecting the backbones) and the (baseplane normals).
            # so the second and third vector b_L, b_N
            rotation_B[:,[1,2]] *= -1
     
        # Extract origins of triads
        origin_A = input_A_[:,0]  # shape (n, 3)
        origin_B = input_B_[:,0]  # shape (n, 3)

        return rotation_A, rotation_B, origin_A, origin_B, original_shape


    def compute_parameters(self, rotation_A, rotation_B, origin_A, origin_B):
        """Calculate the parameters between each base pair and mean reference frames.

        Args:
            rotation_A (ndarray): Rotation matrices of shape (n, 3, 3) for the first triad.
            rotation_B (ndarray): Rotation matrices of shape (n, 3, 3) for the second triad.
            origin_A (ndarray): Origins of shape (n, 3) for the first triad.
            origin_B (ndarray): Origins of shape (n, 3) for the second triad.

        Returns:
            rigid_parameters (ndarray): The parameters of shape (n, 12) representing the relative translation and rotation between each base pair.
            trans_mid (ndarray): The mean translational vector of shape (n, 3) between the triads.
            rotation_mid (ndarray): The mean rotation matrix of shape (n, 3, 3) between the triads.
        """
        
        # Linear interpolation of translations
        trans_mid = 0.5 * (origin_A + origin_B)
    
        # Relative translation
        trans_AB = origin_A - origin_B

        # Get relative rotation matrix of base pair
        rotation_BA = rotation_B.transpose(0,2,1) @ rotation_A  # returns shape (n, 3, 3)

        # Get rotation angles based on  rotation matrices
        rotation_angle_BA = RigidBody.extract_omega_values(rotation_BA)

        # Compute halfway rotation matrix and triad (mid frame)
        rotation_halfway = RigidBody.get_rotation_matrix(rotation_angle_BA * 0.5)

        # Get rotation matrix of base pair (aka mean rotation frame)
        rotation_mid = rotation_B @ rotation_halfway 
        
        # Get transaltional coordinate vector and convert to angstroms
        translational_parameters = np.einsum('ijk,ik->ij',rotation_mid.transpose(0,2,1), trans_AB) * 10

        # Get rotational parameters and convert to degrees
        rotational_parameters = np.rad2deg(np.einsum('ijk,ik->ij', rotation_BA.transpose(0,2,1), rotation_angle_BA))
                
        # Merge translational and rotational parameters
        rigid_parameters = np.hstack((translational_parameters, rotational_parameters))

        # Return the parameters and the mean reference frame
        return rigid_parameters, trans_mid, rotation_mid


    def calculate_parameters(self,frames_A, frames_B, is_step=False):
        """Calculate the parameters between each base pair and mean reference frames.

        Assumes frames are of shape (n_frames, n_residues, 4, 3) where the last two dimensions are the base triads.
        The base triads consist of an origin (first index) and three vectors (latter 3 indices) representing the base frame.
        With the order of the vectors being: b_R, b_L, b_D, b_N.

        Args:
            frames_A (ndarray): Frames of shape (n_frames, n_residues, 4, 3) representing the base triads for chain A.
            frames_B (ndarray): Frames of shape (n_frames, n_residues, 4, 3) representing the base triads for chain B.
            is_step (bool, optional): Flag indicating if the input is a single step or a trajectory. Defaults to False.

        Notes:
            Note the vectors are stored rowwise in the base triads, and not the usual column representation of the rotation matrices.

        Returns:
            params (ndarray): The parameters of shape (n_frames, n_residues, 6) representing the relative translation and rotation between each base pair.
            mean_reference_frames (ndarray): The mean reference frames of shape (n_bp, n_frames, 4, 3) representing the mean reference frame of each base pair.
        """
                
        # Reshape frames
        rotation_A, rotation_B, origin_A, origin_B, original_shape = self.reshape_input(frames_A,frames_B, is_step=is_step)

        # Compute parameters
        if not is_step:
            # Flip from row to column representation of the rotation matrices
            rotation_A = rotation_A.transpose(0,2,1)
            rotation_B = rotation_B.transpose(0,2,1)
            params, mean_origin, mean_rotation = self.compute_parameters(rotation_A, rotation_B, origin_A, origin_B)
        else:
            # Switch the input of the B and A triads to get the correct parameters
            params, mean_origin, mean_rotation = self.compute_parameters(rotation_B, rotation_A, origin_B, origin_A)

        # Reshape the parameters to the original shape
        params = params.reshape(original_shape[0], original_shape[1], 6).swapaxes(0, 1)

        # Collect mean reference frames from mid frames of each base pair
        mean_reference_frames = np.hstack((mean_origin[:, np.newaxis, :],mean_rotation)).reshape(original_shape)

        if is_step:
            # Creating an array of zeros with shape (10000, 1, 6)
            extra_column = np.zeros((params.shape[0], 1, 6))

            # Concatenating the existing array and the extra column along the second axis
            params = np.concatenate((extra_column,params), axis=1)

        # Return the parameters and the mean reference frames
        return  params, mean_reference_frames if not is_step else params


    def analyse_frames(self):
        """Analyze the trajectory and compute parameters."""

        # Get base reference frames for each residue
        frames_A = np.array([self.base_frames[res] for res in self.res_A])
        frames_B = np.array([self.base_frames[res] for res in self.res_B])

        # Compute parameters between each base pair and mean reference frames
        self.bp_params, self.mean_reference_frames = self.calculate_parameters(frames_A, frames_B)
        
        # Extract mean reference frames for each neighboring base pair
        B1_triads = self.mean_reference_frames[:-1] # select all but the last frame
        B2_triads = self.mean_reference_frames[1:] # select all but the first frame

        # Compute parameters between each base pair and mean reference frames
        self.step_params = self.calculate_parameters(B1_triads, B2_triads, is_step=True)[0]

        # Store mean reference frame / aka base pair triads as frames and transpose rotation matrices back to row wise
        self.frames = self.mean_reference_frames
        self.frames[:, :, 1:, :] = np.transpose(self.frames[:, :, 1:, :], axes=(0, 1, 3, 2))
        self._clean_parameters()

    def _clean_parameters(self):
        """Clean the parameters by removing the first and last frame."""
        self.step_parameter_names = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
        self.base_parameter_names = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']
        self.names = self.base_parameter_names + self.step_parameter_names
        self.parameters = np.dstack((self.bp_params, self.step_params))

    def get_parameters(self,step=False,base=False):
        """Return the computed parameters of shape (n_frames, n_base_pairs, n_parameters)"""
        if step and not base:
            return self.step_params, self.step_parameter_names
        elif base and not step:
            return self.bp_params, self.base_parameter_names
        elif not step and not base:
            return self.parameters, self.names
        raise ValueError("Use only one of step=True or base=True, or neither.")
        
    def get_parameter(self,name='twist') -> np.ndarray:
        """Get the parameter of the DNA structure, choose frome the following:
        - shift, slide, rise, tilt, roll, twist, shear, stretch, stagger, buckle, propeller, opening

        Args:
            name (str): parameter name

        Returns:
            parameter(ndarray) : parameter in shape (n_frames, n_base_pairs)"""

        if name not in self.names:
            raise ValueError(f"Parameter {name} not found.")
        return self.parameters[:,:,self.names.index(name)]
    

    def plot_parameters(self, fig=None, ax=None, mean=True, std=True,figsize=[10,3.5], save=False,step=True,base=True,base_color='cornflowerblue',step_color='coral'):
        """Plot the rigid base parameters of the DNA structure
        Args:
            fig: figure
            ax: axis
            mean: plot mean
            std: plot standard deviation
            figsize: figure size
            save: save figure
        Returns:
            figure, axis"""

        import matplotlib.pyplot as plt

        cols = step + base

        if fig is None and ax is None:
            fig,ax = plt.subplots(cols,6, figsize=[12,2*cols])
            ax = ax.flatten()
        if step and not base:
            names = self.step_parameter_names
        elif base and not step:
            names = self.base_parameter_names
        elif base and step:
            names = self.names

        for _,name in enumerate(names):
            if name in self.step_parameter_names:
                color = step_color
            else:
                color = base_color
            para = self.get_parameter(name)
            mean = np.mean(para, axis=0)
            std = np.std(para, axis=0)
            x = range(len(mean))
            #ax[_].errorbar(x,mean, yerr=std, fmt='-', color=color)
            ax[_].fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
            ax[_].plot(mean, color=color,lw=1)    
            ax[_].scatter(x=x,y=mean,color=color,s=10)
            ax[_].set_title(name)

        fig.tight_layout()
        if save:
            fig.savefig('parameters.png')
        return fig, ax 

class SingleStrandFrames(NucleicFrames):
    def __init__(self, traj, chainid=0, fit_reference=False):
        self._init_common(traj, fit_reference=fit_reference) 

        self.chainids = [chainid]
        self.chainid = chainid
        self.residues = self.get_residues(chain_index=chainid, reverse=False)

        self.base_frames = self.get_base_reference_frames()
        self.analyse_frames()

    # Inherits from NucleicFrames:
    # Functions:
    # - get_residues
    # - load_reference_bases
    # - _prepare_reference_fit_data
    # - _get_fitted_base_vectors
    # - get_base_vectors
    # - reshape_input
    # - compute_parameters
    # - calculate_parameters
    # 
    # object inits:
    # - self.traj = traj
    # - self.top = traj.topology
    # - self.fit_reference = fit_reference
    # - self.reference_base_map = {"U": "T"}
    # - self.reference_fit_data = self._prepare_reference_fit_data() if self.fit_reference else {}

    def get_base_reference_frames(self):
        """Get reference frames for each residue in the strand."""
        reference_frames = {}
        for res in self.residues: # in NucleicFrames loop over implied double strand : for res in self.res_A + self.res_B:
            res_traj = self.traj.atom_slice([at.index for at in res.atoms])
            reference_frames[res] = self.get_base_vectors(res_traj)
        return reference_frames 

    def analyse_frames(self):
        """Build per-residue frames and strand-local step parameters."""
        self.step_parameter_names = ["shift", "slide", "rise", "tilt", "roll", "twist"]
        self.base_parameter_names = ["shear", "stretch", "stagger", "buckle", "propeller", "opening"]

        self.frames = np.array([self.base_frames[res] for res in self.residues])

        if len(self.residues) > 1:
            self.step_params = self.calculate_parameters(
                self.frames[:-1], self.frames[1:], is_step=True
            )[0]
        else:
            self.step_params = np.zeros(
                (self.traj.n_frames, len(self.residues), len(self.step_parameter_names))
            )

        self.names = self.step_parameter_names
        self.parameters = self.step_params

    def get_parameters(self, step=False, base=False):
        if base:
            raise NotImplementedError(
                "Base-pair parameters require paired strands. "
                "Single-stranded nucleic acids expose strand-local step parameters only."
            )
        return self.step_params, self.step_parameter_names

    def get_parameter(self, name="twist"):
        if name in self.base_parameter_names:
            raise NotImplementedError(
                "Base-pair parameters require paired strands. "
                "Single-stranded nucleic acids expose strand-local step parameters only."
            )
        if name not in self.step_parameter_names:
            raise ValueError(f"Parameter {name} not found.")
        return self.step_params[:, :, self.step_parameter_names.index(name)]