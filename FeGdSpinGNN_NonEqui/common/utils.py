import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import torch


def _to_numpy(x):
    """Convert torch tensors or arrays to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_system_id(data):
    """Safely extract system_id as an int from a Data object."""
    if not hasattr(data, "system_id"):
        raise AttributeError("Data object has no attribute 'system_id'")
    sid = data.system_id
    if hasattr(sid, "item"):
        sid = sid.item()
    return int(sid)


def _extract_graph_components(data):
    """
    Extract positions, atom_types, and edge_index from a Data object.
    Assumes:
        - positions in data.pos [N, 3]
        - features in data.x, with atom type in data.x[:, 0]
        - edges in data.edge_index [2, E]
    """
    positions  = _to_numpy(data.pos)          # [N, 3]
    atom_types = _to_numpy(data.x[:, 0])      # [N]
    edge_index = _to_numpy(data.edge_index)   # [2, E]
    return positions, atom_types, edge_index


def plot_systems_3d(
    dataset,
    systems=None,
    show_edges=True,
    atom_type_map=None,
    title="3D graphs by system_id"
):
    """
    Plot one 3D subplot per system_id.

    Parameters
    ----------
    dataset : indexable collection of Data
        Your dataset (e.g. dataset_train).
    systems : list[int] or None
        Which system_ids to plot and in what order.
        If None, all unique system_ids in the dataset will be used (sorted).
    show_edges : bool
        Whether to draw edges between atoms.
    atom_type_map : dict or None
        Mapping from atom_type -> {label, color}.
        Default: {0: Gd (blue), 1: Fe (orange)}.
    title : str
        Global figure title.
    """

    # Default atom type mapping
    if atom_type_map is None:
        atom_type_map = {
            0: dict(label="Gd", color="blue"),
            1: dict(label="Fe", color="orange"),
        }

    # Determine systems if not provided
    if systems is None:
        system_ids = []
        for i in range(len(dataset)):
            sid = _get_system_id(dataset[i])
            system_ids.append(sid)
        systems = sorted(set(system_ids))

    systems = list(systems)  # ensure list

    # Collect one sample per system (first occurrence)
    samples = {}
    target_set = set(systems)

    for i in range(len(dataset)):
        data = dataset[i]
        sid = _get_system_id(data)

        if sid in target_set and sid not in samples:
            samples[sid] = data

        if len(samples) == len(target_set):
            break

    # Warn if some systems not found
    missing = [s for s in systems if s not in samples]
    if missing:
        print(f"Warning: no samples found for systems: {missing}")

    # Create subplot grid: 1 row, len(systems) columns
    fig = make_subplots(
        rows=1,
        cols=len(systems),
        specs=[[{"type": "scene"} for _ in systems]],
        subplot_titles=[f"System {s}" for s in systems],
    )

    # Add each system to its column
    for col, system_id in enumerate(systems, start=1):
        if system_id not in samples:
            continue  # skip missing system

        data = samples[system_id]
        positions, atom_types, edge_index = _extract_graph_components(data)

        # --- optional edges ---
        if show_edges:
            edge_x, edge_y, edge_z = [], [], []
            num_edges = edge_index.shape[1]
            for i in range(num_edges):
                s, d = edge_index[0, i], edge_index[1, i]
                edge_x.extend([positions[s, 0], positions[d, 0], None])
                edge_y.extend([positions[s, 1], positions[d, 1], None])
                edge_z.extend([positions[s, 2], positions[d, 2], None])

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="gray", width=1),
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

        # --- atoms by type ---
        for atom_type, cfg in atom_type_map.items():
            label = cfg.get("label", str(atom_type))
            color = cfg.get("color", "gray")

            mask = (atom_types == atom_type)
            if not np.any(mask):
                continue

            fig.add_trace(
                go.Scatter3d(
                    x=positions[mask, 0],
                    y=positions[mask, 1],
                    z=positions[mask, 2],
                    mode="markers",
                    marker=dict(size=3, color=color),
                    name=label if col == 1 else None,   # legend only in first subplot
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

        # Same aspect ratio per subplot
        fig.update_scenes(aspectmode="data", row=1, col=col)

    # Final layout
    fig.update_layout(
        width=500 * len(systems),
        height=600,
        title=title,
    )

    fig.show()


def plot_spin_B_histograms(dataset, systems=None, bins=30):
    """
    For each system_id:
      - aggregate spins (x[:,2:5]) over *all* samples with that system_id
      - aggregate B-fields (y[:,0:3]) over *all* samples with that system_id
      - make separate matplotlib histograms for Sx,Sy,Sz and Bx,By,Bz
    """

    # Determine which systems to use
    if systems is None:
        all_ids = [_get_system_id(dataset[i]) for i in range(len(dataset))]
        systems = sorted(set(all_ids))
    systems = list(systems)
    wanted = set(systems)

    # Dicts: sid -> list of arrays to concat later
    spins_dict = {sid: [] for sid in systems}
    bfield_dict = {sid: [] for sid in systems}

    # Collect data from ALL graphs/timesteps
    for i in range(len(dataset)):
        data = dataset[i]
        sid = _get_system_id(data)
        if sid not in wanted:
            continue

        x_feat = _to_numpy(data.x)       # [N_atoms, 5]
        y_val  = _to_numpy(data.y)       # [N_atoms, 3] in your case

        spins = x_feat[:, 2:5]           # Sx, Sy, Sz

        if y_val.ndim == 1:
            y_val = np.tile(y_val.reshape(1, 3), (spins.shape[0], 1))
        else:
            y_val = y_val[:, :3]         # Bx, By, Bz

        spins_dict[sid].append(spins)
        bfield_dict[sid].append(y_val)

    # Now plot per system, using all collected values
    for sid in systems:
        if not spins_dict[sid]:
            print(f"Warning: no data for system {sid}")
            continue

        spins_all = np.concatenate(spins_dict[sid], axis=0)
        b_all     = np.concatenate(bfield_dict[sid], axis=0)

        # --- Figure 1: spins ---
        fig_s, axes_s = plt.subplots(1, 3, figsize=(12, 4))
        for j, comp in enumerate(["Sx", "Sy", "Sz"]):
            ax = axes_s[j]
            ax.hist(spins_all[:, j], bins=bins)
            ax.set_title(comp)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        fig_s.suptitle(f"Spin histograms – System {sid}")
        fig_s.tight_layout()
        plt.show()

        # --- Figure 2: B field ---
        fig_b, axes_b = plt.subplots(1, 3, figsize=(12, 4))
        for j, comp in enumerate(["Bx", "By", "Bz"]):
            ax = axes_b[j]
            ax.hist(b_all[:, j], bins=bins)
            ax.set_title(comp)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
        fig_b.suptitle(f"B-field histograms – System {sid}")
        fig_b.tight_layout()
        plt.show()

import torch
import matplotlib.pyplot as plt


def plot_correlation(model, loader, device, normalizer):
    """Plot predicted vs true B-field values (denormalized via normalizer)."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)  # [N, 3]

            # denormalize
            out_denorm = normalizer.unnormalize_target(out.detach().cpu())
            y_denorm = normalizer.unnormalize_target(batch.y.detach().cpu())

            all_preds.append(out_denorm)
            all_targets.append(y_denorm)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ["Bx", "By", "Bz"]

    for i, comp in enumerate(components):
        axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.3, s=1)

        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], "k--")

        axes[i].set_xlabel("True")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(comp)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_true_pred_B_fields(model, dataset, sample_indices, normalizer,
                            scale_B=50000, figsize=(7, 7), device='cpu'):
    """
    Plot true and predicted B-fields for each sample in separate figures
    """
    model.eval()

    with torch.no_grad():
        for sample_idx in sample_indices:
            data = dataset[sample_idx].to(device)
            pred = model(data)

            # denormalize B-fields
            pred_denorm = normalizer.unnormalize_target(pred.cpu())
            true_denorm = normalizer.unnormalize_target(data.y.cpu())

            pos = data.pos.cpu().numpy()

            plt.figure(figsize=figsize)
            # true
            plt.quiver(pos[:, 0][::3], pos[:, 2][::3], true_denorm[:, 0][::3], true_denorm[:, 2][::3],
                      color='red', scale=scale_B, label='B-field (True)',
                      alpha=0.7, width=0.004)

            # predicted
            plt.quiver(pos[:, 0][::3], pos[:, 2][::3], pred_denorm[:, 0][::3], pred_denorm[:, 2][::3],
                      color='blue', scale=scale_B, label='B-field (Pred)',
                      alpha=0.6, width=0.003)

            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(f'Sample {sample_idx}: True vs Predicted B-field')
            plt.legend()
            plt.grid(True, alpha=0.8)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

def training_plot(num_epochs, train_losses, val_losses):
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_physical_metrics(model, loader, device, target_normalizer):
    """

    Args:
        model: trained model
        loader: PyG DataLoader
        device: torch device
        target_normalizer: fitted TargetNormalizer instance

    Returns:
        dict with MSE, RMSE, MAE (total and per-component)
    """

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Model outputs are normalized vectors
            out_norm = model(batch)

            # Denormalize using scalar magnitude std
            out = target_normalizer.unnormalize_target(out_norm)
            y = target_normalizer.unnormalize_target(batch.y)

            all_preds.append(out)
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)     # [N_total, 3]
    all_targets = torch.cat(all_targets, dim=0) # [N_total, 3]

    diff = all_preds - all_targets

    # Global metrics
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(diff))

    # Per-component MSE
    mse_per_comp = torch.mean(diff ** 2, dim=0)

    return {
        'mse': mse.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'mse_x': mse_per_comp[0].item(),
        'mse_y': mse_per_comp[1].item(),
        'mse_z': mse_per_comp[2].item(),
    }




