# Visualisation des résultats après entraînement
# Trois choses ici : les courbes d'apprentissage, la matrice de confusion
# et une grille d'exemples pour voir visuellement ce que le modèle rate

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


# Affichage des images correctement après la normalisation ImageNet
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def plot_learning_curves(history: dict, output_dir: str) -> None:
    sns.set_style("whitegrid")
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Époque où la val accuracy est la meilleure (pour annoter les graphes)
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc   = max(history["val_acc"])
    best_loss  = history["val_loss"][best_epoch - 1]

    COLOR_TRAIN = "#2563EB"   
    COLOR_VAL   = "#DC2626"   
    COLOR_BEST  = "#16A34A"  

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        "Courbes d'apprentissage – Détection de Défauts de Surface (NEU)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Graphe 1 : Loss
    ax_loss.plot(epochs, history["train_loss"], color=COLOR_TRAIN, linewidth=2,
                 marker="o", markersize=4, label="Train")
    ax_loss.plot(epochs, history["val_loss"],   color=COLOR_VAL,   linewidth=2,
                 marker="o", markersize=4, label="Validation")

    # Zone de gap entre train et val (plus le gap est large, plus on surfit)
    ax_loss.fill_between(epochs, history["train_loss"], history["val_loss"],
                         alpha=0.08, color=COLOR_VAL)

    # Ligne et annotation sur la meilleure époque
    ax_loss.axvline(x=best_epoch, color=COLOR_BEST, linewidth=1.5,
                    linestyle="--", alpha=0.7, label=f"Meilleure époque ({best_epoch})")
    ax_loss.annotate(
        f"val loss\n{best_loss:.4f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + 0.5, best_loss + 0.05),
        fontsize=8, color=COLOR_BEST,
        arrowprops=dict(arrowstyle="->", color=COLOR_BEST, lw=1.2),
    )

    ax_loss.set_title("Perte (Cross-Entropy Loss)", fontsize=12, fontweight="semibold")
    ax_loss.set_xlabel("Époque", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.legend(fontsize=9)

    # Graphe 2 : Accuracy
    ax_acc.plot(epochs, history["train_acc"], color=COLOR_TRAIN, linewidth=2,
                marker="o", markersize=4, label="Train")
    ax_acc.plot(epochs, history["val_acc"],   color=COLOR_VAL,   linewidth=2,
                marker="o", markersize=4, label="Validation")

    ax_acc.fill_between(epochs, history["train_acc"], history["val_acc"],
                        alpha=0.08, color=COLOR_VAL)

    ax_acc.axvline(x=best_epoch, color=COLOR_BEST, linewidth=1.5,
                   linestyle="--", alpha=0.7, label=f"Meilleure époque ({best_epoch})")
    ax_acc.annotate(
        f"{best_acc:.2f}%",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 0.5, best_acc - 3),
        fontsize=8, color=COLOR_BEST,
        arrowprops=dict(arrowstyle="->", color=COLOR_BEST, lw=1.2),
    )

    ax_acc.set_title("Précision (Accuracy)", fontsize=12, fontweight="semibold")
    ax_acc.set_xlabel("Époque", fontsize=10)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.legend(fontsize=9)

    plt.tight_layout()
    _save_and_show(fig, output_dir, "learning_curves.png")
    sns.set_style("darkgrid") 


def evaluate_model(
    model:       nn.Module,
    val_loader:  DataLoader,
    class_names: list[str],
    device:      str,
    output_dir:  str,
) -> None:
    all_preds, all_labels = _predict_all(model, val_loader, device)

    cm            = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Matrice de Confusion – Défauts NEU (Validation)", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (ax_abs,  cm,            "Comptages absolus",                  "d"),
        (ax_norm, cm_normalized, "Taux normalisés (par ligne réelle)", ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    _save_and_show(fig, output_dir, "confusion_matrix.png")

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n" + "─" * 62)
    print("  RAPPORT DE CLASSIFICATION (Validation)")
    print("─" * 62)
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Rapport sauvegardé : {report_path}")


def visualize_predictions(
    model:       nn.Module,
    val_loader:  DataLoader,
    class_names: list[str],
    device:      str,
    output_dir:  str,
    n_images:    int = 16,
) -> None:
    # récupération des n premières images de la validation avec leurs prédictions
    model.eval()
    images_list, preds_list, labels_list = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for img, pred, label in zip(images, preds, labels):
                images_list.append(img)
                preds_list.append(pred.item())
                labels_list.append(label.item())
                if len(images_list) >= n_images:
                    break
            if len(images_list) >= n_images:
                break

    cols = 4
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Exemples de Prédictions  (rouge = erreur)", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n_images:
            ax.axis("off")
            continue
        ax.imshow(_denormalize(images_list[i]))
        pred_name = class_names[preds_list[i]]
        true_name = class_names[labels_list[i]]
        color     = "green" if pred_name == true_name else "red"
        ax.set_title(f"Pred : {pred_name}\nRéel : {true_name}", color=color, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    _save_and_show(fig, output_dir, "sample_predictions.png")


def compare_models(
    histories:   dict[str, dict],
    models:      dict[str, nn.Module],
    val_loader:  DataLoader,
    class_names: list[str],
    device:      str,
    output_dir:  str,
) -> None:
    """
    Génère un graphe comparatif entre le CNN from scratch et MobileNetV2.
    histories = {"CNN Scratch": history1, "MobileNetV2": history2}
    models    = {"CNN Scratch": model1,   "MobileNetV2": model2}
    """
    COLORS = {"CNN Scratch": "#F59E0B", "MobileNetV2": "#2563EB"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Comparaison des Modèles — CNN Scratch vs Transfer Learning (MobileNetV2)",
        fontsize=13, fontweight="bold",
    )

    # Courbes Accuracy 
    ax = axes[0]
    for name, history in histories.items():
        epochs = range(1, len(history["val_acc"]) + 1)
        ax.plot(epochs, history["val_acc"], color=COLORS[name], linewidth=2.5,
                marker="o", markersize=4, label=name)
    ax.set_title("Val Accuracy par époque")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)

    # Courbes Loss
    ax = axes[1]
    for name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        ax.plot(epochs, history["val_loss"], color=COLORS[name], linewidth=2.5,
                marker="o", markersize=4, label=name)
    ax.set_title("Val Loss par époque")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)

    # Barres Accuracy finale 
    ax = axes[2]
    noms   = list(histories.keys())
    scores = [max(h["val_acc"]) for h in histories.values()]
    bars   = ax.bar(noms, scores, color=[COLORS[n] for n in noms], width=0.4, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Meilleure Val Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Ligne de référence : performance d'un inspecteur humain (~72%)
    ax.axhline(72, color="#DC2626", linewidth=1.5, linestyle="--", alpha=0.7,
               label="Inspecteur humain (~72%)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save_and_show(fig, output_dir, "model_comparison.png")

    # Résumé texte
    print("\n" + "─" * 62)
    print("  COMPARAISON DES MODÈLES")
    print("─" * 62)
    for name in noms:
        acc = max(histories[name]["val_acc"])
        epo = len(histories[name]["val_acc"])
        print(f"  {name:<22} Accuracy : {acc:.2f}%  |  Époques : {epo}")
    gagnant = max(histories, key=lambda n: max(histories[n]["val_acc"]))
    print(f"\n  Meilleur modèle : {gagnant}")
    print("─" * 62 + "\n")


def _predict_all(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    # Inverse la normalisation pour pouvoir afficher l'image normalement
    img = tensor.permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0, 1)


def _save_and_show(fig: plt.Figure, output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Sauvegardé : {path}")
