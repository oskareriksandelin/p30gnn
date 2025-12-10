def create_nequip_dataset(data_root='data', output_file='fegd_dataset.xyz', systems=None):
    """
    Convert FeGd magnetic data to ASE extended XYZ format for NequIP training.
    
    Args:
        data_root (str): Root directory containing FeGd_data_POSCAR_* folders
        output_file (str): Output filename for the dataset
        systems (list): Which systems to include (e.g., [2,3,4]). None = all
    
    Returns:
        List of ASE Atoms objects
    """
    import os
    import numpy as np
    import pandas as pd
    from ase import Atoms
    from ase.io import write
    
    if systems is None:
        systems = [2]
    
    all_structures = []
    
    for sys in systems:
        path = os.path.join(data_root, f'FeGd_data_POSCAR_{sys}')
        
        # Load data
        pos = pd.read_csv(os.path.join(path, 'coord.FeGd_100.out'), sep=r'\s+', header=None,
                          names=['id','x','y','z','i1','i2'])[['x','y','z']]
        
        B = pd.read_csv(os.path.join(path, 'befftot.FeGd_100.out'), sep=r'\s+', header=None, comment='#',
                        names=['Iter','Site','Replica','B_x','B_y','B_z','B'])
        
        m = pd.read_csv(os.path.join(path, 'moment.FeGd_100.out'), sep=r'\s+', engine='python',
                        header=None, comment='#',
                        names=['iter','ens','Site','|Mom|','M_x','M_y','M_z'])
        
        # Get unique timesteps
        timesteps = B['Iter'].unique()
        
        for t in timesteps:
            # Filter by timestep
            m_t = m[m['iter'] == t].sort_values('Site')
            B_t = B[B['Iter'] == t].sort_values('Site')
            
            # Create atom symbols (24 Gd + 76 Fe per 100 atoms, repeated 8 times)
            symbols = []
            for i in range(8):
                symbols.extend(['Gd'] * 24 + ['Fe'] * 76)
            
            # Create ASE Atoms object
            atoms = Atoms(
                symbols=symbols,
                positions=pos.values,
                pbc=True,  # Assume periodic boundary conditions
                cell=np.eye(3) * 10  # Default cubic cell, adjust as needed
            )
            
            # Attach properties as arrays (NequIP reads these as training targets)
            atoms.info['B_field'] = B_t[['B_x','B_y','B_z']].values  # Per-atom target
            atoms.arrays['moments'] = m_t[['M_x','M_y','M_z']].values  # Per-atom features
            atoms.info['energy'] = B_t['B_x'].sum()  # Scalar target (sum of B field)
            
            all_structures.append(atoms)
    
    # Write to extended XYZ format
    write(output_file, all_structures)
    print(f"Dataset created: {output_file} with {len(all_structures)} structures")
    
    return all_structures
    
if __name__ == "__main__":
    import sys
    import os
    
    # Determine paths relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, "..", "data"))
    output_file = os.path.join(data_root, "fegd_dataset.xyz")
    
    # Ensure data directory exists
    os.makedirs(data_root, exist_ok=True)
    
    print(f"Creating NequIP dataset from: {data_root}")
    print(f"Output file: {output_file}")
    
    try:
        structures = create_nequip_dataset(
            data_root=data_root,
            output_file=output_file,
            systems=[2]
        )
        print(f"\n✓ Successfully created dataset with {len(structures)} structures")
    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        sys.exit(1)


