import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset


def parse_nml(path):
    """
    Parse a .nml file to extract features as a dictionary.
    Args:
        path (str): Path to the file
    Returns:
        dict: {feature_key: feature_value_array}
    """
    feats = {}
    with open(path) as f:
        for line in f:
            if '=' in line and not line.strip().startswith('&'):
                key, val = line.split('=', 1)
                nums = re.findall(r"[-+]?\d*\.\d+E?[+-]?\d*|\d+", val)
                if nums:
                    feats[key.strip()] = np.array(nums, dtype=float)
    return feats


def build_edges_from_neighbors(nbr_data, edge_features):
    """
    Build edge_index and edge_attr tensors from neighbor DataFrame.

    Args:
        nbr_data (pd.DataFrame): DataFrame with neighbor info, must contain columns:
            'iatom', 'jatom', 'dx', 'dy', 'dz', 'rij'
        features (list): List of feature column names to include in edge_attr
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

    # currently hardcoded features
    # TODO: make this flexible if needed
    edge_attr_np = nbr_data[edge_features].to_numpy(dtype=np.float32)

    # Zero-copy conversion to torch
    edge_index = torch.from_numpy(edge_index_np)
    edge_attr = torch.from_numpy(edge_attr_np)

    return edge_index, edge_attr


def extract_static_tensor(static_feats, n_atoms):
    """
    Convert static features dictionary to tensor
    Args:
        static_feats (dict): {atom_id: {feature_key: feature_value_array}}
        n_atoms (int): Number of atoms in the system
    """

    # TODO: make feature keys flexible if needed
    feature_keys = ['valence', 'lmax'] # ex for now
    rows = []
    for i in range(1, n_atoms + 1):
        feats = static_feats.get(i, {})
        row = []
        for k in feature_keys:
            val = feats.get(k, np.array([0.0])) # default to 0.0 if missing
            # convert to flat array for both scalar and array features
            val = np.asarray(val, dtype=float).ravel()
            # add all components
            row.extend(val.tolist())
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float)


class FeGdMagneticDataset(Dataset):
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
    """
    
    def __init__(self, root, systems=[2, 3, 4, 5, 6, 7, 8, 9], cutoff_dist=None, edge_features='all', use_static_features=False):
        self.root = root
        self.systems = systems
        self.use_static_features = use_static_features
        self.cutoff_dist = cutoff_dist
        self.edge_features = edge_features
        
        self.cache = {}
        self.index_map = []

        self._load_all_systems()

    def _load_all_systems(self):
        for sys in tqdm(self.systems, desc="Loading systems"):
            path = os.path.join(self.root, f'FeGd_data_POSCAR_{sys}')

            pos = pd.read_csv(f'{path}/coord.FeGd_100.out', sep=r'\s+', header=None,
                              names=['id','x','y','z','i1','i2'])[['x','y','z']]

            B = pd.read_csv(f'{path}/befftot.FeGd_100.out', sep=r'\s+', header=None, comment='#',
                            names=['Iter','Site','Replica','B_x','B_y','B_z','B'])

            m = pd.read_csv(f'{path}/moment.FeGd_100.out', sep=r'\s+', engine='python',
                            header=None, comment='#',
                            names=['iter','ens','Site','|Mom|','M_x','M_y','M_z'])

            nbr = pd.read_csv(f'{path}/struct.FeGd_100.out', sep=r'\s+', comment='#',
                          names=['iatom','jatom','itype','jtype','dx','dy','dz','Jij','rij'])

            static = None
            if self.use_static_features:
                static = self._load_static_features(sys)

            # cache data so we don't reload every time
            self.cache[sys] = dict(pos=pos, B=B, m=m, nbr=nbr, static=static)
            
            # build index map for (system, timestep)
            n_steps = B['Iter'].nunique() # unique timesteps
            for t in range(n_steps):
                self.index_map.append((sys, t))

    def _load_static_features(self, system):
        nml_dir = os.path.join(self.root, 'RSLMTO', f'POSCAR_{system}', 'Cont')
        # Load and parse all .nml files in the directory. 
        # Note: sorted in reverse order to match atom indexing with Gd first.
        files = sorted([f for f in os.listdir(nml_dir) if f.startswith(('Fe', 'Gd'))], reverse=True)
        return {i+1: parse_nml(os.path.join(nml_dir, f)) for i, f in enumerate(files)}

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sys, t = self.index_map[idx]
        data_cache = self.cache[sys]

        pos_df = data_cache['pos']
        B_df = data_cache['B']
        m_df = data_cache['m']
        nbr_df = data_cache['nbr']
        static = data_cache['static']

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
        m_t = m_df[m_df['iter'] == t].sort_values('Site')
        B_t = B_df[B_df['Iter'] == t].sort_values('Site')

        moment = torch.tensor(m_t[['M_x','M_y','M_z']].values, dtype=torch.float)
        y = torch.tensor(B_t[['B_x','B_y','B_z']].values, dtype=torch.float)

        x_parts = [
            node_type,
            moment
        ]

        if static is not None:
            x_parts.append(extract_static_tensor(static, n_atoms))

        x = torch.cat(x_parts, dim=1)

        # edges
        if self.cutoff_dist: # apply cutoff
            nbr_df = nbr_df[nbr_df['rij'] <= self.cutoff_dist]
            
        edge_index, edge_attr = build_edges_from_neighbors(nbr_df, self.edge_features)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            system_id=sys,
            timestep=t
        )
    

if __name__ == "__main__":
    from time import time
    start = time()
    dataset = FeGdMagneticDataset(
        root=r'data',
        systems=[2],
        cutoff_dist=0.3,  # example cutoff distance
        edge_features='ALL',  # example edge features
        use_static_features=False, # probaly not needed for now  
    )
    print(f"Dataset loaded in {time() - start:.2f} seconds")
    
    print(f"Dataset size: {len(dataset)}")
    print(f"\nFirst graph:")
    time_start = time()
    data = dataset[0]
    print(f"Graph loaded in {time() - time_start:.2f} seconds")
    print("Graph info for first data point:")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}, ")
    print(f"Target shape: {data.y.shape}")
    print(f"System ID: {data.system_id}")
    print(f"Timestep: {data.timestep}")
    print(f"First atom node features:\n{data.x[0]}")
    print(f"First neighbor edge features:\n{data.edge_attr[0]}")