# Deux architectures disponibles :
#   - CNN from scratch : construit bloc par bloc
#   - MobileNetV2     : Transfer Learning, pré-entraîné sur ImageNet

import torch.nn as nn
import torch.optim as optim
from torchvision import models


# CNN FROM SCRATCH

class CNNScratch(nn.Module):
    """
    CNN construit de zéro avec 4 blocs convolutifs.

    Architecture :
        [Conv2D(32)  + BN + ReLU + MaxPool + Dropout] x1
        [Conv2D(64)  + BN + ReLU + MaxPool + Dropout] x1
        [Conv2D(128) + BN + ReLU + MaxPool + Dropout] x1
        [Conv2D(256) + BN + ReLU + MaxPool + Dropout] x1
        → AdaptiveAvgPool → Flatten
        → Dense(512) + ReLU + Dropout → Dense(num_classes)

    Le BatchNorm accélère la convergence et stabilise l'entraînement
    L'AdaptiveAvgPool évite de calculer la taille exacte après les convolutions
    """

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 : détecte les features simples (bords, contours)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.1),

            # Bloc 2 : features intermédiaires (textures)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.15),

            # Bloc 3 : patterns complexes propres aux défauts
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            # Bloc 4 : représentations de haut niveau
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),
        )

        # Pooling global pour vecteur fixe de 256 éléments quelle que soit la taille d'entrée
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_cnn_scratch(num_classes: int, dropout_rate: float, device: str) -> nn.Module:
    model = CNNScratch(num_classes, dropout_rate).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres entraînables : {total:,} / {total:,} (100.0%)\n")

    return model


# TRANSFER LEARNING (MobileNetV2)

def build_mobilenet(num_classes: int, dropout_rate: float, device: str) -> nn.Module:
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # On bloque tous les paramètres du backbone pour la phase 1
    for param in backbone.parameters():
        param.requires_grad = False

    in_features = backbone.classifier[1].in_features  

    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate / 2),
        nn.Linear(512, num_classes),
    )

    model = backbone.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres entraînables : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")

    return model


def unfreeze_backbone(model: nn.Module, fine_tune_lr: float) -> optim.Optimizer:
    # On ne dégèle que les 5 derniers blocs, pas tout le réseau.
    # Les premières couches détectent des features génériques (bords, couleurs)
    # qu'on n'a pas besoin de modifier.
    for block in list(model.features.children())[-5:]:
        for param in block.parameters():
            param.requires_grad = True

    # LR différentiel : très faible sur le backbone pour ne pas "oublier" ImageNet,
    # un peu plus élevé sur la tête qui doit encore s'adapter.
    optimizer = optim.Adam([
        {"params": model.features.parameters(),   "lr": fine_tune_lr},
        {"params": model.classifier.parameters(), "lr": fine_tune_lr * 5},
    ])

    print("  Fine-tuning activé : 5 derniers blocs MobileNetV2 dégelés.\n")
    return optimizer
