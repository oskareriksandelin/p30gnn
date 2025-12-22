import torch
import torch.nn.functional as F
from tqdm import tqdm
import time


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def mae_loss(pred, target):
    return F.l1_loss(pred, target)


class SimpleTrainer:
    """
    Minimal trainer for non-equivariant GNNs where model(batch) -> [N, 3].
    Designed to be dead-simple and easy to extend later.
    """

    def __init__(self, model, device, optimizer, scheduler=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            pred = self.model(batch)      # [N, 3]
            target = batch.y              # [N, 3]
            loss = mse_loss(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(self.device)

            pred = self.model(batch)
            target = batch.y
            loss = mse_loss(pred, target)

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def fit(self, train_loader, val_loader, epochs):
        best_val_loss = float("inf")
        scheduler = self.scheduler
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss   = self.evaluate(val_loader)

            print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f} | lr {self.optimizer.param_groups[0]['lr']:.6f}")
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")




class BFieldTrainer:
    """
    Trainer for GNN B-field prediction (B_x, B_y, B_z per atom).
    
    Prints ONE line per epoch.
    No per-batch output.
    """

    def __init__(self, model, device, loss_fn=mse_loss, lr=1e-3, use_tqdm=False):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.use_tqdm = use_tqdm

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_mse": []
        }

        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        num_atoms = 0

        pbar = tqdm(loader, leave=False, desc="Training", ascii=True)

        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(
                x=batch.x,
                pos=batch.pos,
                edge_index=batch.edge_index,
                batch=batch.batch,
            )

            # pred: (N, 3), batch.y: (N, 3)
            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch.x.size(0)  # multiply by number of atoms
            num_atoms += batch.x.size(0)

            # tqdm live update
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_atoms, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()

        total_loss = 0
        total_mae = 0
        total_mse = 0
        num_atoms = 0

        iterator = tqdm(loader, leave=False, desc="Evaluating") if self.use_tqdm else loader

        for batch in iterator:
            batch = batch.to(self.device)

            pred = self.model(
                x=batch.x,
                pos=batch.pos,
                edge_index=batch.edge_index,
                batch=batch.batch,
            )

            # pred: (N, 3), batch.y: (N, 3)
            target = batch.y

            loss = self.loss_fn(pred, target)
            mae  = F.l1_loss(pred, target)
            mse  = F.mse_loss(pred, target)

            total_loss += loss.item() * batch.x.size(0)
            total_mae  += mae.item()  * batch.x.size(0)
            total_mse  += mse.item()  * batch.x.size(0)
            num_atoms  += batch.x.size(0)

        num_atoms = max(num_atoms, 1)

        return {
            "loss": total_loss / num_atoms,
            "mae":  total_mae  / num_atoms,
            "mse":  total_mse  / num_atoms,
        }

    def train(self, train_loader, val_loader, epochs, patience=10):

        print("=" * 80)
        print("Starting training")
        print(f"Early stopping after {patience} epochs without improvement")
        print("=" * 80)

        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_loss = val_metrics["loss"]
            val_mae  = val_metrics["mae"]
            val_mse  = val_metrics["mse"]

            # Save
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_mse"].append(val_mse)

            # Improvement?
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                epochs_no_improve  = 0
                marker = "[best]"
            else:
                epochs_no_improve += 1
                marker = f"({epochs_no_improve}/{patience})"

            # ONE LINE per epoch
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train {train_loss:7.4f} | "
                f"Val {val_loss:7.4f} | "
                f"MAE {val_mae:7.4f} | "
                f"MSE {val_mse:7.4f} | "
                f"{marker}"
            )

            if epochs_no_improve >= patience:
                print("=" * 80)
                print("Early stopping")
                print(f"Best epoch: {self.best_epoch}")
                print(f"Best val loss: {self.best_val_loss:.6f}")
                print("=" * 80)
                break

        return self.history