# Chargement et prétraitement des images

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Ces valeurs viennent d'ImageNet et sont imposées par MobileNetV2
# Normalisation par ces chiffres, sinon les poids pré-entraînés donnent n'importe quoi
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    # Pour la validation, on ne fait aucune augmentation, seulement le resize et la normalisation
    # L'augmentation est réservée au train pour éviter le surapprentissage
    base = [transforms.Resize((img_size, img_size))]

    if is_train:
        base += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]

    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    return transforms.Compose(base)


def load_datasets(config: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    pin = config["device"] == "cuda"

    train_ds = datasets.ImageFolder(
        root=config["train_dir"],
        transform=build_transforms(config["img_size"], is_train=True),
    )
    val_ds = datasets.ImageFolder(
        root=config["val_dir"],
        transform=build_transforms(config["img_size"], is_train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=pin,
    )

    class_names = train_ds.classes
    print(f"  Classes détectées ({len(class_names)}) : {class_names}")
    print(f"  Train : {len(train_ds)} images  |  Validation : {len(val_ds)} images\n")

    return train_loader, val_loader, class_names
