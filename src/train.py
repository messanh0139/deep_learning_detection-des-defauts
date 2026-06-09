# Boucle d'entraînement et callbacks.
# train_model accepte n'importe quel nn.Module — CNN scratch ou MobileNetV2.
# Le flag two_phase active le fine-tuning (Transfer Learning uniquement).

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import unfreeze_backbone


class EarlyStopping:
    """Arrête l'entraînement si la val loss ne bouge plus."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False


def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device:    str,
    is_train:  bool,
) -> tuple[float, float]:
    # Même fonction pour train et val, le flag is_train contrôle
    # model.train/eval et l'activation des gradients
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(
    config:       dict,
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    save_path:    str,
    two_phase:    bool = False,
) -> tuple[nn.Module, dict]:
    """
    Entraîne le modèle passé en paramètre et sauvegarde les meilleurs poids.

    two_phase=True  → Transfer Learning : phase 1 (tête seule) puis fine-tuning backbone.
    two_phase=False → CNN scratch : entraînement direct de tous les paramètres.
    """
    device    = config["device"]
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stop = EarlyStopping(config["patience"], config["min_delta"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    phase        = 1
    fine_tuned   = False

    _print_phase_header(1 if two_phase else 0)

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, False)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

        _log_epoch(epoch, config["num_epochs"], time.time() - t0,
                   train_loss, train_acc, val_loss, val_acc, is_best)

        if early_stop(val_loss):
            print(f"\n  Early Stopping à l'époque {epoch} (phase {phase}).")

            if two_phase and not fine_tuned:
                model.load_state_dict(best_weights)
                optimizer  = unfreeze_backbone(model, config["fine_tune_lr"])
                scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2
                )
                early_stop.reset()
                fine_tuned = True
                phase      = 2
                _print_phase_header(2)
            else:
                break

    model.load_state_dict(best_weights)
    print(f"\n  Meilleure val accuracy : {best_val_acc:.2f}%")
    print(f"  Modèle sauvegardé      : {save_path}\n")

    return model, history


def _print_phase_header(phase: int) -> None:
    labels = {
        0: "Entraînement du CNN from Scratch",
        1: "PHASE 1 – Entraînement de la tête de classification (MobileNetV2)",
        2: "PHASE 2 – Fine-tuning des couches profondes (MobileNetV2)",
    }
    print(f"\n{'─'*62}")
    print(f"  {labels[phase]}")
    print(f"{'─'*62}")


def _log_epoch(
    epoch: int, total: int, elapsed: float,
    tl: float, ta: float, vl: float, va: float,
    is_best: bool,
) -> None:
    marker = "  ✓ MEILLEUR" if is_best else ""
    print(
        f"  Époque {epoch:02d}/{total}  [{elapsed:4.0f}s]  "
        f"Train Loss {tl:.4f}  Acc {ta:5.2f}%  │  "
        f"Val Loss {vl:.4f}  Acc {va:5.2f}%"
        + marker
    )
