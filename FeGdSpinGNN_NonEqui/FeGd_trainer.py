import torch
import torch.nn.functional as F
from tqdm import tqdm
import time


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def mae_loss(pred, target):
    return F.l1_loss(pred, target)

import math
import torch


def cosine_angle_loss(B_pred, B_true, mag_threshold=1e-3, eps=1e-8):
    mag_pred = torch.norm(B_pred, dim=1)
    mag_true = torch.norm(B_true, dim=1)

    valid = mag_true > mag_threshold
    if not valid.any():
        return torch.tensor(0.0, device=B_pred.device)

    dot = torch.sum(B_pred[valid] * B_true[valid], dim=1)
    denom = mag_pred[valid] * mag_true[valid] + eps
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    return (1.0 - cos).mean()


@torch.no_grad()
def magnitude_and_angle_error(B_pred, B_true, mag_threshold=1e-3, eps=1e-8):

    # --- magnitudes ---
    mag_pred = torch.norm(B_pred, dim=1)
    mag_true = torch.norm(B_true, dim=1)

    # magnitude MAE (always meaningful)
    mag_mae = torch.mean(torch.abs(mag_pred - mag_true)).item()

    # --- angle masking ---
    valid = mag_true > mag_threshold
    if not valid.any():
        return mag_mae, float("nan")

    # --- cosine of angle ---
    dot = torch.sum(B_pred[valid] * B_true[valid], dim=1)
    denom = mag_pred[valid] * mag_true[valid] + eps
    cos = torch.clamp(dot / denom, -1.0, 1.0)

    # --- angle in degrees ---
    ang_deg = (torch.acos(cos) * (180.0 / math.pi)).mean().item()
    return mag_mae, ang_deg

import torch.nn as nn
from tqdm import tqdm


import torch
import torch.nn as nn
from tqdm import tqdm


class BFieldTrainer:
    def __init__(self, model, device, optimizer, lambda_angle=0.0, eps=1e-8, mag_threshold=1e-3, scheduler=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_angle = lambda_angle
        self.eps = eps
        self.mag_threshold = mag_threshold
        self.criterion = nn.MSELoss()

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_mse": [],
            "val_angle": [],
            "lr": [],
        }

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(batch)
            target = batch.y

            mse = self.criterion(pred, target)
            angle = cosine_angle_loss(pred, target, mag_threshold=self.mag_threshold, eps=self.eps)

            loss = mse + self.lambda_angle * angle
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        preds, targets = [], []

        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(self.device)

            pred = self.model(batch)
            target = batch.y

            mse = self.criterion(pred, target)
            angle = cosine_angle_loss(pred, target, mag_threshold=self.mag_threshold, eps=self.eps)
            loss = mse + self.lambda_angle * angle

            total_loss += loss.item()
            total_mse += mse.item()

            preds.append(pred.cpu())
            targets.append(target.cpu())

        avg_loss = total_loss / max(1, len(loader))
        avg_mse = total_mse / max(1, len(loader))

        B_pred = torch.cat(preds)
        B_true = torch.cat(targets)
        mag_mae, ang_deg = magnitude_and_angle_error(B_pred, B_true, mag_threshold=self.mag_threshold, eps=self.eps)

        self.history["val_loss"].append(avg_loss)
        self.history["val_mse"].append(avg_mse)
        self.history["val_mae"].append(mag_mae)
        self.history["val_angle"].append(ang_deg)

        return avg_loss, mag_mae, avg_mse, ang_deg

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, mag_mae, val_mse, val_ang = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]['lr'])


            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train loss {train_loss:.6f} | "
                f"Val loss {val_loss:.6f} | "
                f"MSE {val_mse:.6f} | "
                f"Angle {val_ang:.2f}° | "
                f"|B| MAE {mag_mae:.6f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.6f}"
            )
        return self.history


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