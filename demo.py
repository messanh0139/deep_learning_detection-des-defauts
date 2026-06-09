# Script de démonstration

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import CONFIG
from src.model  import build_mobilenet

# Paramètres de la démo

N_IMAGES    = 12    # nombre d'images à afficher
RANDOM_SEED = 7   

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# Noms plus lisibles pour l'affichage 
LABELS_FR = {
    "crazing":        "Craquelures",
    "inclusion":      "Inclusion",
    "patches":        "Taches",
    "pitted_surface": "Piqûres",
    "rolled-in_scale":"Calamine",
    "scratches":      "Rayures",
}


def load_model(model_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Lance d'abord l'entraînement avec : python main.py"
        )
    model = build_mobilenet(num_classes=len(CLASSES), dropout_rate=CONFIG["dropout_rate"], device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collect_images(val_dir: str, n: int, seed: int) -> list[tuple[str, str]]:
    # Collecte n images aléatoires réparties sur toutes les classes
    all_images = []
    for cls in CLASSES:
        folder = os.path.join(val_dir, cls)
        files  = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
        all_images.extend([(path, cls) for path in files])

    random.seed(seed)
    return random.sample(all_images, min(n, len(all_images)))


def predict(model, image_path: str, transform, device: str, threshold: float) -> tuple[str, str, float]:
    """
    Retourne : (statut, type_defaut, confiance)
      statut     : "Saine" ou "Défectueuse"
      type_defaut: classe prédite (ou "—" si saine)
      confiance  : probabilité max du modèle
    """
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    confidence, idx = probs.max(dim=0)
    confidence_val  = confidence.item()

    if confidence_val < threshold:
        # Le modèle n'est pas assez sûr → surface considérée comme saine
        return "Saine", "—", confidence_val
    else:
        return "Défectueuse", CLASSES[idx.item()], confidence_val


def run_demo():
    device = CONFIG["device"]
    print(f"\n{'='*58}")
    print("  DÉMONSTRATION — Détection de Défauts de Surface NEU")
    print(f"  Device : {device.upper()}")
    print(f"{'='*58}\n")

    print("  Chargement du modèle...")
    model     = load_model(CONFIG["best_model_path"], device)
    transform = get_transform(CONFIG["img_size"])
    print("  Modèle chargé.\n")

    samples   = collect_images(CONFIG["val_dir"], N_IMAGES, RANDOM_SEED)
    threshold = CONFIG["confidence_threshold"]

    print(f"  Seuil de confiance (Saine/Défectueuse) : {threshold*100:.0f}%\n")

    # Prédictions
    results = []
    correct = 0
    for path, true_cls in samples:
        statut, pred_cls, confidence = predict(model, path, transform, device, threshold)
        is_ok = (statut == "Défectueuse") and (pred_cls == true_cls)
        correct += int(is_ok)
        results.append((path, true_cls, statut, pred_cls, confidence, is_ok))

        status_icon = "✓" if is_ok else "✗"
        type_affiche = LABELS_FR.get(pred_cls, pred_cls)
        print(f"  {status_icon}  Réel : {LABELS_FR[true_cls]:<14}  "
              f"Statut : {statut:<13}  "
              f"Type : {type_affiche:<14}  "
              f"Confiance : {confidence*100:.1f}%")

    accuracy = 100 * correct / len(results)
    print(f"\n  Accuracy sur cet échantillon : {accuracy:.1f}%  ({correct}/{len(results)})\n")

    # Visualisation
    _plot_results(results)


def _plot_results(results: list) -> None:
    cols = 4
    rows = (len(results) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.8))
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        "Détection Automatisée de Défauts de Surface — NEU Dataset\n"
        "Modèle : MobileNetV2 (Transfer Learning)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for i, ax in enumerate(axes.flat):
        if i >= len(results):
            ax.axis("off")
            continue

        path, true_cls, statut, pred_cls, confidence, is_ok = results[i]

        img = np.array(Image.open(path).convert("RGB"))
        ax.imshow(img)

        # Bordure verte = correct, rouge = erreur, orange = détecté sain
        if statut == "Saine":
            color = "#F59E0B"
        else:
            color = "#16A34A" if is_ok else "#DC2626"

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        type_affiche = LABELS_FR.get(pred_cls, pred_cls)
        title = (
            f"Réel : {LABELS_FR[true_cls]}\n"
            f"{statut} — {type_affiche}\n"
            f"Confiance : {confidence*100:.1f}%"
        )
        ax.set_title(title, fontsize=8, color=color, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

    # Légende globale
    patch_ok   = mpatches.Patch(color="#16A34A", label="Défectueuse — correct")
    patch_err  = mpatches.Patch(color="#DC2626", label="Défectueuse — erreur")
    patch_sain = mpatches.Patch(color="#F59E0B", label="Considérée saine")
    fig.legend(handles=[patch_ok, patch_err, patch_sain], loc="lower center",
               ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    save_path = os.path.join(CONFIG["output_dir"], "demo_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Visualisation sauvegardée : {save_path}\n")


if __name__ == "__main__":
    run_demo()
