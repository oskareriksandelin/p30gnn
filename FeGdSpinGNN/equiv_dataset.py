# @title
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset
from e3nn import o3
import math



def equiv_build_edges_from_neighbors(nbr_data, edge_features, lmax=2):
    """
    Build edge_index and edge_attr tensors from neighbor DataFrame.

    Args:
        nbr_data (pd.DataFrame): DataFrame with neighbor info, must contain columns:
            'iatom', 'jatom', 'dx', 'dy', 'dz', 'rij'
        edge_features (list or str): List of edge feature names to include, or 'all' to include all available features.
            'dx', 'dy', 'dz': relative position components
            'rij': distance between atoms
    Returns:
        edge_index (torch.LongTensor): Tensor of shape [2, num_edges]
        edge_attr (torch.FloatTensor): Tensor of shape [num_edges, num_edge_features]
    """

    valid_features = nbr_data.drop('Jij', axis=1).columns   #defines valid features excluding 'Jij'

    if isinstance(edge_features, str) and edge_features.lower() == 'all':
        edge_features = ['dx', 'dy', 'dz', 'rij']
    elif any(feat not in valid_features for feat in edge_features):
        raise ValueError("One or more specified edge features are not present in the neighbor data columns.")
    elif edge_features is None or not isinstance(edge_features, list):
        raise ValueError("edge_features must be 'all' or a list containing the correct feature names (see docstring)")

    edge_index_np = nbr_data[['iatom', 'jatom']].to_numpy().T
    edge_index_np = edge_index_np.astype(np.int64)
    edge_index_np -= 1  # zero-based indexing

    # Extract edge attributes
    edge_attr_np = nbr_data[edge_features].to_numpy(dtype=np.float32)

    # convert edge vectors to unit vectors if dx, dy, dz are included
    if edge_features == ['dx', 'dy', 'dz', 'rij']:
        vecs = edge_attr_np[:, :3]
        unit_vecs = vecs / edge_attr_np[:, 3:4]  # divide by rij to get unit vector
        edge_attr_np[:, :3] = unit_vecs

        edge_sh = o3.spherical_harmonics(
            l=list(range(lmax + 1)),
            x=torch.from_numpy(unit_vecs),
            normalize=True,
            normalization="component"
        )

    # Zero-copy conversion to torch
    edge_index = torch.from_numpy(edge_index_np)
    edge_attr = torch.from_numpy(edge_attr_np)
    return edge_index, edge_attr, edge_sh




class EquivFeGdMagneticDataset(Dataset):
    """
    PyTorch Geometric Dataset for FeGd magnetic systems.

    Each timestep is treated as a separate graph.
    Magnetic field (B_x, B_y, B_z) is the target variable.
    Node features include spin moments (M_x, M_y, M_z), and optionally with static features from .nml files.
    Edge features are relative position vectors between atoms.

    Args:
        root (str): Root directory containing FeGd_data_POSCAR_* folders
        systems (list): List of system numbers to include (e.g., [2, 3, 4])
        cutoff_dist (float): cutoff distance for edge construction (None to use all neighbors in struct file)
        use_static_features (bool): Whether to include .nml static features
        edge_features (str or list): 'all' to use all available edge features, or list of specific features to include e.g ['dx', 'dy', 'rij']
                \n 'dx', 'dy', 'dz': relative position components
                \n 'rij': distance between atoms
        transform_rotate (callable): Optional transformation function for data rotation augmentation
    """

    def __init__(self, root, systems=[2, 3, 4, 5, 6, 7, 8, 9], lmax=2, cutoff_dist=None, edge_features='all', transform_rotate=None):
        self.root = root
        self.systems = systems
        self.cutoff_dist = cutoff_dist
        self.edge_features = edge_features
        self.transform_rotate = transform_rotate
        self.lmax = lmax

        self.cache = {}
        self.index_map = []

        self._load_all_systems()

    def _load_all_systems(self):
        for sys in tqdm(self.systems, desc="Loading systems"):
            path = os.path.join(self.root, f'FeGd_data_POSCAR_{sys}')
            path_b_data = os.path.join(self.root, f'fields/POSCAR_{sys}')

            pos = pd.read_csv(f'{path}/coord.FeGd_100.out', sep=r'\s+', header=None,
                              names=['id','x','y','z','i1','i2'])[['x','y','z']]

            B = pd.read_csv(f'{path_b_data}/bintefftot.FeGd_100.out', sep=r'\s+', header=None, comment='#',
                            names=['Iter','Site','Replica','B_x','B_y','B_z', 'B', 'sld_x', 'sld_y', 'sld_z', 'sld'])

            m = pd.read_csv(f'{path}/moment.FeGd_100.out', sep=r'\s+', engine='python',
                            header=None, comment='#',
                            names=['Iter','ens','Site','|Mom|','M_x','M_y','M_z'])

            nbr = pd.read_csv(f'{path}/struct.FeGd_100.out', sep=r'\s+', comment='#',
                          names=['iatom','jatom','itype','jtype','dx','dy','dz','Jij','rij'])


            # cache data so we don't reload every time
            self.cache[sys] = dict(pos=pos, B=B, m=m, nbr=nbr)

            # build index map for (system, timestep)
            iter_vals = sorted(B['Iter'].unique())  # use real Iter values from the file
            for t in iter_vals:
                self.index_map.append((sys, t))


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sys, t = self.index_map[idx]
        data_cache = self.cache[sys]

        pos_df = data_cache['pos']
        B_df = data_cache['B']
        m_df = data_cache['m']
        nbr_df = data_cache['nbr']

        pos = torch.tensor(pos_df.values, dtype=torch.float)
        n_atoms = len(pos)

        # node type one-hot encoding
        node_type = torch.zeros((n_atoms, 2), dtype=torch.float)
        fe = torch.tensor([1.0, 0.0])
        gd = torch.tensor([0.0, 1.0])
        for i in range(8):
            start = i * 100
            node_type[start:start+24] = gd
            node_type[start+24:start+100] = fe

        # timestep filtering (sort to ensure correct order)
        m_t = m_df[m_df['Iter'] == t].sort_values('Site')
        B_t = B_df[B_df['Iter'] == t].sort_values('Site')

        #moment = torch.tensor(m_t[['M_x','M_y','M_z']].values, dtype=torch.float)
        spin = torch.tensor(
            m_t[['M_x','M_y','M_z']].values,
            dtype=torch.float
        )  # [N, 3]
        spin_mag = torch.norm(spin, dim=1, keepdim=True)  # [N, 1]
        spin_dir = spin / (spin_mag + 1e-8)  # [N, 3]
        y = torch.tensor(B_t[['B_x','B_y','B_z']].values, dtype=torch.float)

        # edges
        if self.cutoff_dist: # apply cutoff
            nbr_df = nbr_df[nbr_df['rij'] <= self.cutoff_dist]

        if self.transform_rotate is None:
            edge_index, edge_attr, edge_sh = equiv_build_edges_from_neighbors(nbr_df, self.edge_features, lmax=self.lmax)
        else:
            edge_index, edge_attr, _ = equiv_build_edges_from_neighbors(nbr_df, self.edge_features, lmax=self.lmax)

        edge_dist = edge_attr[:, 3:4]

        # Apply augmentation if specified
        rel_pos = edge_attr[:, :3] #placeholder for relative position vector
        if self.transform_rotate is not None:
            spin_dir, y, pos, edge_vec = self.transform_rotate(spin_dir, y, pos, rel_pos)
            edge_sh = o3.spherical_harmonics(
                l=list(range(self.lmax + 1)),
                x=edge_vec,
                normalize=True,
                normalization="component"
            )

        x_parts = [
            node_type,   # 2x0e
            spin_mag,    # 1x0e
            spin_dir     # 1x1o
        ]
        x = torch.cat(x_parts, dim=1)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_dist,
            edge_sh=edge_sh,
            y=y,
            pos=pos
        )