# Détection automatisée de défauts de surface — Dataset NEU
# KODJO Messanh Yaovi | Deep Learning Appliqué en Entreprise
#
# Lancement :
#   source .venv/bin/activate
#   python main.py

import os
from src.config   import CONFIG
from src.data     import load_datasets, build_transforms
from src.model    import build_cnn_scratch, build_mobilenet
from src.train    import train_model
from src.evaluate import (plot_learning_curves, evaluate_model,
                          visualize_predictions, compare_models)
from src.roi      import plot_roi
from torchvision  import datasets
from torch.utils.data import DataLoader


def main() -> None:
    _print_banner()

    # ── 1. Données ────────────────────────────────────────────────────────────
    print("[ 1/6 ]  Chargement des données")
    train_loader, val_loader, class_names = load_datasets(CONFIG)
    num_classes = len(class_names)

    # val_loader propre (ordre déterministe) pour l'évaluation
    clean_val_loader = _make_val_loader()

    # ── 2. CNN from Scratch ───────────────────────────────────────────────────
    print("[ 2/6 ]  Entraînement — CNN from Scratch")
    cnn_scratch = build_cnn_scratch(num_classes, CONFIG["dropout_rate"], CONFIG["device"])
    path_scratch = os.path.join(CONFIG["output_dir"], "cnn_scratch.pth")

    model_scratch, history_scratch = train_model(
        config=CONFIG,
        model=cnn_scratch,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=path_scratch,
        two_phase=False,   # pas de fine-tuning, tout s'entraîne depuis le début
    )
    plot_learning_curves(history_scratch, CONFIG["output_dir"])

    # ── 3. MobileNetV2 (Transfer Learning) ───────────────────────────────────
    print("[ 3/6 ]  Entraînement — MobileNetV2 (Transfer Learning)")
    mobilenet = build_mobilenet(num_classes, CONFIG["dropout_rate"], CONFIG["device"])

    model_mobilenet, history_mobilenet = train_model(
        config=CONFIG,
        model=mobilenet,
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=CONFIG["best_model_path"],
        two_phase=True,    # phase 1 tête → phase 2 fine-tuning backbone
    )
    plot_learning_curves(history_mobilenet, CONFIG["output_dir"])

    # ── 4. Comparaison des deux modèles ───────────────────────────────────────
    print("[ 4/6 ]  Comparaison CNN Scratch vs MobileNetV2")
    compare_models(
        histories={"CNN Scratch": history_scratch, "MobileNetV2": history_mobilenet},
        models={"CNN Scratch": model_scratch, "MobileNetV2": model_mobilenet},
        val_loader=clean_val_loader,
        class_names=class_names,
        device=CONFIG["device"],
        output_dir=CONFIG["output_dir"],
    )

    # ── 5. Évaluation détaillée du meilleur modèle ───────────────────────────
    print("[ 5/6 ]  Évaluation détaillée — MobileNetV2")
    evaluate_model(model_mobilenet, clean_val_loader, class_names,
                   CONFIG["device"], CONFIG["output_dir"])
    visualize_predictions(model_mobilenet, clean_val_loader, class_names,
                          CONFIG["device"], CONFIG["output_dir"])

    # ── 6. Analyse ROI ────────────────────────────────────────────────────────
    print("[ 6/6 ]  Analyse de l'impact ROI")
    best_accuracy = max(history_mobilenet["val_acc"])
    plot_roi(best_accuracy, CONFIG["output_dir"])

    _print_footer(CONFIG["output_dir"])


def _make_val_loader() -> DataLoader:
    val_ds = datasets.ImageFolder(
        root=CONFIG["val_dir"],
        transform=build_transforms(CONFIG["img_size"], is_train=False),
    )
    return DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)


def _print_banner() -> None:
    print("\n" + "=" * 62)
    print("  Détection Automatisée de Défauts de Surface — NEU Dataset")
    print("  KODJO Messanh Yaovi  |  Deep Learning Appliqué en Entreprise")
    print("=" * 62)
    print(f"  Device : {CONFIG['device'].upper()}")
    print(f"  Images : {CONFIG['img_size']}×{CONFIG['img_size']} px  "
          f"|  Batch : {CONFIG['batch_size']}  |  Époques max : {CONFIG['num_epochs']}")
    print("  Modèles : CNN from Scratch  +  MobileNetV2 (Transfer Learning)")
    print("=" * 62 + "\n")


def _print_footer(output_dir: str) -> None:
    print("\n" + "=" * 62)
    print("  Entraînement terminé. Fichiers générés dans :", output_dir)
    print("    • cnn_scratch.pth           poids CNN from scratch")
    print("    • best_model.pth            poids MobileNetV2")
    print("    • learning_curves.png       courbes Loss & Accuracy")
    print("    • model_comparison.png      comparaison des deux modèles")
    print("    • confusion_matrix.png      matrice de confusion")
    print("    • sample_predictions.png    exemples de prédictions")
    print("    • classification_report.txt rapport F1 par classe")
    print("    • roi_analysis.png          analyse ROI sur 3 ans")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
