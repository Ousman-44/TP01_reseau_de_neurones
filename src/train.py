"""Entraîne le détecteur, évalue (IoU moyen, % IoU > 0.5) et sauvegarde une figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.detector import Detector, ShapeDataset, compute_loss, iou, predict


def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne."""
    model.train() # Permet de passer le modèle en mode entraînement (active dropout, batchnorm, etc.)
    total_loss = 0.0
    for images,targets in loader :
        optimizer.zero_grad() #réinitialise gradient
        outputs = model(images)  #passe les img dans le modèle pour avoir prédiction
        loss = compute_loss(outputs, targets) #calcule perte entre prediction et cible
        loss.backward() #calcule gradients
        optimizer.step() #met a jour les poids du modèle
        total_loss +=loss.item() #accumule perte pour clacul moyenne
    # faites un backward et un step de l'optimizer.
    
    
    return total_loss / len(loader.dataset)


def evaluate(model: Detector, loader: DataLoader, size: int) -> tuple[float, float]:
    model.eval()
    ious = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            for logit, target in zip(logits, targets):
                box, _ = predict(logit, size)
                row, col = divmod(int(target[0].argmax()), grid)
                _, dx, dy, w, h = target[:, row, col].tolist()
                cx, cy = (col + dx) * size / grid, (row + dy) * size / grid
                truth = torch.tensor([cx - w * size / 2, cy - h * size / 2,
                                      cx + w * size / 2, cy + h * size / 2])
                ious.append(float(iou(box, truth[None])[0]))
    t = torch.tensor(ious)
    return t.mean().item(), (t > 0.5).float().mean().item()

