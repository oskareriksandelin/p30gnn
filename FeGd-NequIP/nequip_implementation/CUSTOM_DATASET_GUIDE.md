# Custom Dataset Integration Guide for NequIP

This guide walks you through the complete process of creating a custom dataset and integrating it into NequIP with a `config.yml` file.

## Overview

NequIP uses a modular architecture where:
1. **Datasets** inherit from `AtomicDataset` and provide raw atomic data
2. **DataModules** handle data loading, batching, and splitting
3. **Transforms** convert raw data into the format NequIP models expect
4. **Config files** orchestrate everything using Hydra instantiation

---

## Step 1: Understand the Required Data Format

NequIP uses `AtomicDataDict` to standardize atomic structure data. Your dataset must provide dictionaries with these **required fields**:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `pos` or `POSITIONS_KEY` | torch.Tensor | `[N_atoms, 3]` | Atomic coordinates (Cartesian) |
| `atomic_numbers` or `ATOMIC_NUMBERS_KEY` | torch.LongTensor | `[N_atoms]` | Atomic numbers (1=H, 6=C, 8=O, etc.) |

**Optional fields** (commonly used):
- `energy` / `TOTAL_ENERGY_KEY`: Total energy (scalar)
- `forces` / `FORCE_KEY`: Atomic forces `[N_atoms, 3]`
- `stress` / `STRESS_KEY`: Stress tensor `[3, 3]`
- `cell` / `CELL_KEY`: Periodic cell `[3, 3]`
- `pbc` / `PBC_KEY`: Periodic boundary conditions `[3]` (bool)

Key constants are defined in `/nequip/data/AtomicDataDict.py`.

---

## Step 2: Create Your Custom Dataset Class

Create a new file in `/nequip/data/dataset/` or in your own module:

```python
# my_custom_dataset.py
import torch
from typing import Union, List, Dict, Any, Callable
from nequip.data import AtomicDataDict
from nequip.data.dataset.base_datasets import AtomicDataset
from nequip.data.dict import from_dict


class MyCustomDataset(AtomicDataset):
    """Custom dataset for [YOUR DATA SOURCE].
    
    Args:
        file_path (str): Path to your data file
        transforms (List[Callable]): List of data transforms to apply
        key_mapping (Dict[str, str]): Map your data keys to AtomicDataDict keys
        **kwargs: Additional arguments specific to your data format
    """
    
    def __init__(
        self,
        file_path: str,
        transforms: List[Callable] = [],
        key_mapping: Dict[str, str] = None,
        **kwargs
    ):
        super().__init__(transforms=transforms)
        self.file_path = file_path
        
        # Set default key mapping if not provided
        if key_mapping is None:
            key_mapping = {
                "positions": AtomicDataDict.POSITIONS_KEY,
                "atomic_numbers": AtomicDataDict.ATOMIC_NUMBERS_KEY,
                "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
                "forces": AtomicDataDict.FORCE_KEY,
            }
        self.key_mapping = key_mapping
        
        # Load your data - store it as a list of dictionaries
        self.data_list: List[AtomicDataDict.Type] = []
        self._load_data()
    
    def _load_data(self):
        """Load data from file and convert to AtomicDataDict format."""
        # Example: Load from your custom format
        # This is pseudo-code - adapt to your data format
        raw_data = self._read_your_data_format(self.file_path)
        
        for frame in raw_data:
            # Create a dictionary with your data
            data_dict = {
                AtomicDataDict.POSITIONS_KEY: torch.tensor(
                    frame['positions'], dtype=torch.float32
                ),
                AtomicDataDict.ATOMIC_NUMBERS_KEY: torch.tensor(
                    frame['atomic_numbers'], dtype=torch.long
                ),
                # Add other fields as available
            }
            
            # If you have energy/forces
            if 'energy' in frame:
                data_dict[AtomicDataDict.TOTAL_ENERGY_KEY] = torch.tensor(
                    frame['energy'], dtype=torch.float32
                )
            if 'forces' in frame:
                data_dict[AtomicDataDict.FORCE_KEY] = torch.tensor(
                    frame['forces'], dtype=torch.float32
                )
            
            # Convert to proper format and store
            self.data_list.append(from_dict(data_dict))
    
    def _read_your_data_format(self, file_path: str):
        """Override this to read your specific data format."""
        # Example implementations:
        # - Load from HDF5: h5py.File()
        # - Load from numpy: np.load()
        # - Load from custom format: your_parser()
        # - Load from directories: glob.glob()
        raise NotImplementedError("Implement data loading for your format")
    
    def __len__(self) -> int:
        """Return total number of frames in dataset."""
        return len(self.data_list)
    
    def _get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        """Return data for requested indices."""
        if isinstance(indices, slice):
            return self.data_list[indices]
        else:
            return [self.data_list[i] for i in indices]
```

### Key Implementation Details:

- **Inherit from `AtomicDataset`**: Provides the required interface
- **Implement `__len__()`**: Return dataset size
- **Implement `_get_data_list()`**: Return raw data for given indices
- **Use `from_dict()`**: Converts raw dicts to proper tensor format
- **Use `AtomicDataDict` constants**: Ensures consistent key naming
- **Store transforms**: Parent class applies them automatically

---

## Step 3: Register Your Dataset (Optional but Recommended)

If you want your dataset discoverable by Hydra:

### Option A: Add to official module
Edit `/nequip/data/dataset/__init__.py`:
```python
from .my_custom_dataset import MyCustomDataset

__all__ = [
    # ... existing imports ...
    "MyCustomDataset",
]
```

### Option B: Use full module path in config
In your config (Step 5), use the full path:
```yaml
_target_: my_package.my_custom_dataset.MyCustomDataset
```

---

## Step 4: Prepare Your Data

Before creating the config, ensure your data is in the correct format:

### Example: Converting from common formats

**From NumPy arrays:**
```python
import numpy as np

# Your data
positions = np.random.randn(10, 3)  # 10 atoms, 3 coordinates
atomic_numbers = np.array([6, 1, 1, 1, 1, 6, 8, 8, 7, 6])  # 10 atoms
energy = 42.5
forces = np.random.randn(10, 3)

data_dict = {
    AtomicDataDict.POSITIONS_KEY: positions,
    AtomicDataDict.ATOMIC_NUMBERS_KEY: atomic_numbers,
    AtomicDataDict.TOTAL_ENERGY_KEY: energy,
    AtomicDataDict.FORCE_KEY: forces,
}
```

**From ASE Atoms objects:**
```python
import ase.io
from nequip.data.ase import from_ase

atoms = ase.io.read("structure.xyz")
data_dict = from_ase(atoms)
```

**From JSON/YAML:**
```python
import json
import yaml

with open("data.json") as f:
    raw_data = json.load(f)  # Should have structure like shown above
```

---

## Step 5: Create Your config.yml

Create a configuration file to use your dataset:

```yaml
# config.yml
# =========
#     RUN
# =========
run: [train, test]

# =========
#     DATA
# =========
data:
  _target_: nequip.data.datamodule.NequIPDataModule
  seed: 123
  
  # Option 1: Use your custom dataset directly
  split_dataset:
    # Use your custom dataset class
    _target_: my_package.my_custom_dataset.MyCustomDataset
    # or if registered in nequip.data.dataset:
    # _target_: nequip.data.dataset.MyCustomDataset
    
    file_path: /path/to/your/data.xyz  # or .npz, .hdf5, etc.
    
    # Optional: custom key mapping if needed
    # key_mapping:
    #   your_pos_key: pos
    #   your_energy_key: energy
    
    # Split into train/val/test
    train: 0.8
    val: 0.1
    test: 0.1
  
  # Option 2: Pre-split datasets (if you have separate files)
  # train_dataset:
  #   _target_: nequip.data.dataset.MyCustomDataset
  #   file_path: /path/to/train_data.xyz
  # val_dataset:
  #   _target_: nequip.data.dataset.MyCustomDataset
  #   file_path: /path/to/val_data.xyz
  # test_dataset:
  #   _target_: nequip.data.dataset.MyCustomDataset
  #   file_path: /path/to/test_data.xyz
  
  # Transforms: convert raw data to model-ready format
  transforms:
    # Map chemical species to model atom types
    - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
      model_type_names: [C, H, O, N]  # Adjust to your atomic species
    
    # Build neighbor list for the model
    - _target_: nequip.data.transforms.NeighborListTransform
      r_max: 5.0  # Cutoff radius in Ångströms
  
  # DataLoader configuration
  train_dataloader:
    batch_size: 5
    num_workers: 4
    shuffle: true
  
  val_dataloader:
    batch_size: 10
    num_workers: 2
    shuffle: false
  
  test_dataloader:
    batch_size: 10
    num_workers: 2
    shuffle: false


# ===========
#    MODEL
# ===========
model:
  _target_: nequip.models.NequIPModel
  r_max: 5.0
  num_layers: 3
  l_max: 1
  num_features: 32
  
  # Must match your chemical species
  chemical_species:
    - C
    - H
    - O
    - N


# ===============
#    TRAINING
# ===============
trainer:
  _target_: nequip.train.lightning.NequIPLightningModule
  max_epochs: 50
  
  # Loss function
  loss:
    _target_: nequip.train.metrics.WeightedHuberLoss
    error_weight: 1.0
    force_weight: 1.0  # Weight for force predictions
  
  # Learning rate scheduler
  scheduler:
    _target_: torch.optim.lr_scheduler.ExponentialLR
    gamma: 0.99
  
  # Checkpointing
  default_root_dir: ./results
  
  # Monitored metric for early stopping
  monitored_metric: val0_epoch/weighted_sum
```

### Important Configuration Notes:

1. **`_target_`**: Full Python path to the class to instantiate (Hydra syntax)
2. **`model_type_names`**: Must match your atomic species in transforms
3. **`r_max`**: Cutoff radius (must be consistent across data/model)
4. **`transforms`**: Applied in order; order matters!
5. **`seed`**: For reproducibility

---

## Step 6: Validate and Test

Before training, test your setup:

```bash
cd /path/to/nequip

# Test dataset loading
python -c "
from nequip.data.dataset import MyCustomDataset
from nequip.data.dict import from_dict

dataset = MyCustomDataset(file_path='path/to/your/data')
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print(f'Positions shape: {sample[\"pos\"].shape}')
"

# Test config parsing and data loading
python -m nequip.train.train --config config.yml --mode test-only
```

---

## Step 7: Start Training

```bash
python -m nequip.train.train --config config.yml
```

---

## Common Dataset Examples

### Example 1: NPZ Format (NumPy)
```python
class NPZCustomDataset(AtomicDataset):
    def _load_data(self):
        data = np.load(self.file_path)
        # Assuming NPZ has: 'positions', 'atomic_numbers', 'energies', 'forces'
        for i in range(data['positions'].shape[0]):
            self.data_list.append(from_dict({
                AtomicDataDict.POSITIONS_KEY: data['positions'][i],
                AtomicDataDict.ATOMIC_NUMBERS_KEY: data['atomic_numbers'],
                AtomicDataDict.TOTAL_ENERGY_KEY: data['energies'][i],
                AtomicDataDict.FORCE_KEY: data['forces'][i],
            }))
```

### Example 2: HDF5 Format
```python
class HDF5CustomDataset(AtomicDataset):
    def _load_data(self):
        with h5py.File(self.file_path, 'r') as f:
            n_frames = f['positions'].shape[0]
            atomic_nums = f['atomic_numbers'][()]  # (N_atoms,)
            
            for i in range(n_frames):
                self.data_list.append(from_dict({
                    AtomicDataDict.POSITIONS_KEY: f['positions'][i],
                    AtomicDataDict.ATOMIC_NUMBERS_KEY: atomic_nums,
                    AtomicDataDict.TOTAL_ENERGY_KEY: f['energies'][i],
                    AtomicDataDict.FORCE_KEY: f['forces'][i],
                }))
```

### Example 3: Directory of Individual Structures
```python
class DirectoryDataset(AtomicDataset):
    def _load_data(self):
        import glob
        files = sorted(glob.glob(f"{self.file_path}/*.xyz"))
        
        for fname in files:
            atoms = ase.io.read(fname)
            self.data_list.append(from_ase(atoms))
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" | Ensure dataset is in Python path or use full module path in config |
| "pos not in data" | Check that `POSITIONS_KEY` is set correctly in your data_dict |
| "Shape mismatch" | Verify positions are `[N_atoms, 3]`, not `[3, N_atoms]` |
| "Tensor dtype error" | Use `torch.float32` for positions/forces, `torch.long` for atomic numbers |
| "Key not recognized" | Import constants from `nequip.data.AtomicDataDict` |
| "Out of memory" | Reduce `batch_size` or `num_workers` in dataloader config |

---

## Summary Checklist

- [ ] Data is in format: positions `[N, 3]`, atomic_numbers `[N]`
- [ ] Created custom dataset class inheriting from `AtomicDataset`
- [ ] Implemented `__len__()` and `_get_data_list()`
- [ ] Used `from_dict()` to convert raw data
- [ ] Created `config.yml` with proper `_target_` paths
- [ ] Set correct `model_type_names` and `chemical_species`
- [ ] Tested dataset loading before training
- [ ] `r_max` is consistent between data transforms and model
- [ ] All required fields present in data dictionaries

---

## References

- NequIP Data Module: `/nequip/data/datamodule/_base_datamodule.py`
- AtomicDataDict: `/nequip/data/AtomicDataDict.py`
- Built-in Datasets: `/nequip/data/dataset/`
- Tutorial Config: `/configs/tutorial.yaml`
