# Tous les hyperparamètres au même endroit

import os
import torch

CONFIG = {
    # Chemins vers les données et les sorties
    "train_dir":        "NEU-DET/train/images",
    "val_dir":          "NEU-DET/validation/images",
    "output_dir":       "outputs",
    "best_model_path":  "outputs/best_model.pth",

    # Les images NEU font déjà 200x200
    "img_size":         200,
    "batch_size":       32,

    # 20 époques max, mais en pratique l'early stopping arrête avant
    "num_epochs":       20,
    "learning_rate":    1e-3,   # pour la tête uniquement (phase 1)
    "fine_tune_lr":     1e-4,   # pour ne pas écraser les poids pré-entraînés
    "dropout_rate":     0.4,

    # Early stopping : on attend 5 époques sans amélioration avant de lâcher
    "patience":         5,
    "min_delta":        1e-4,

    # Seuil de confiance pour la décision Saine / Défectueuse.
    # Si le modèle est moins sûr que ça, on considère la surface comme saine
    # ou du moins non classifiable comme défectueuse connue.
    "confidence_threshold": 0.65,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
