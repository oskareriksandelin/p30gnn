import matplotlib.pyplot as plt
import torch

# Correlation plot
def plot_correlation(model, loader, device, y_mean, y_std):
    """Plot predicted vs true B-field values"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            out_denorm = (out * y_std.to(device) + y_mean.to(device)).cpu()
            y_denorm = (batch.y * y_std.to(device) + y_mean.to(device)).cpu()

            all_preds.append(out_denorm)
            all_targets.append(y_denorm)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['Bx', 'By', 'Bz']

    for i, comp in enumerate(components):
        axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.3, s=1)

        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect')

        axes[i].set_xlabel(f'True')
        axes[i].set_ylabel(f'Predicted')
        axes[i].set_title(f'{comp}')
        #axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def training(num_epochs, train_losses, val_losses):
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()