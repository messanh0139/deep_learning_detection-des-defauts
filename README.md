# Détection Automatisée de Défauts de Surface 

Projet réalisé dans le cadre du cours **Deep Learning Appliqué en Entreprise**.

L'objectif est de construire un système capable de classifier automatiquement des défauts de surface sur des pièces en acier, à partir d'images issues du dataset NEU-DET. Deux approches sont comparées : un CNN entraîné de zéro et un MobileNetV2 par transfer learning. Le tout est conclu par une analyse de l'impact économique en contexte industriel.


## Le dataset

Le dataset utilisé est **NEU-DET**, un benchmark standard en contrôle qualité industriel. Il contient des images en niveaux de gris de surfaces d'acier laminé à chaud, réparties en 6 types de défauts :

| Classe | Description |
|---|---|
| `crazing` | Craquelures — réseau de fines fissures en surface |
| `inclusion` | Inclusions — particules étrangères intégrées au métal |
| `patches` | Taches — zones de coloration irrégulière |
| `pitted_surface` | Piqûres — petits cratères en surface |
| `rolled-in_scale` | Calamine — résidus d'oxyde laminés |
| `scratches` | Rayures — traces linéaires |

Chaque classe contient 300 images (200×200 px), soit 1800 images au total. Le split train/validation est organisé dans `NEU-DET/train/` et `NEU-DET/validation/`.


## Les deux modèles

### CNN from Scratch

Un réseau convolutif construit, sans poids pré-entraînés. Il suit une architecture classique avec 4 blocs progressifs :

```
Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.10)
Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.15)
Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.20)
Conv2D(256) → BN → ReLU → MaxPool → Dropout(0.25)
→ AdaptiveAvgPool → Flatten → Dense(512) → Dense(6)
```

Le Dropout augmente progressivement pour forcer le réseau à ne pas dépendre d'un sous-ensemble de neurones. L'`AdaptiveAvgPool` évite d'avoir à calculer la taille exacte après les convolutions.

### MobileNetV2 — Transfer Learning

On part d'un backbone MobileNetV2 pré-entraîné sur ImageNet et on remplace sa tête de classification par une tête adaptée aux 6 classes NEU.

L'entraînement se fait en **deux phases** :

- **Phase 1** — le backbone est gelé, seule la nouvelle tête s'entraîne (lr = `1e-3`). Ça permet d'adapter rapidement les dernières couches sans écraser les features génériques déjà apprises.
- **Phase 2 (fine-tuning)** — déclenchée automatiquement après l'early stopping de la phase 1. Les 5 derniers blocs du backbone sont dégelés avec un learning rate très faible (`1e-4`) pour affiner les représentations spécifiques aux défauts acier sans "oublier" ImageNet.

---

## Entraînement

Les deux modèles passent par la même boucle d'entraînement avec les mêmes mécanismes de régularisation :

- **Early stopping** : arrêt si la val loss ne descend pas d'au moins `1e-4` en 5 époques
- **ReduceLROnPlateau** : le learning rate est divisé par 2 si pas d'amélioration en 3 époques
- **Sauvegarde automatique** des meilleurs poids à chaque amélioration de la val accuracy

Les images d'entraînement passent par des augmentations géométriques (flips, rotation ±15°) et colorimétriques (luminosité, contraste) pour limiter le surapprentissage. La normalisation ImageNet est appliquée dans les deux cas, même pour le CNN scratch, pour faciliter la comparaison.


## Résultats

| Modèle | Val Accuracy | Remarque |
|---|---|---|
| CNN from Scratch | ~85-90% | entraîné depuis zéro |
| **MobileNetV2** | **98.89%** | transfer learning + fine-tuning |
| Inspecteur humain | ~72% | baseline industrielle |

Rapport de classification final (MobileNetV2, 360 images de validation) :

```
                 precision    recall  f1-score   support

        crazing     1.0000    1.0000    1.0000        60
      inclusion     0.9828    0.9500    0.9661        60
        patches     1.0000    1.0000    1.0000        60
 pitted_surface     1.0000    1.0000    1.0000        60
rolled-in_scale     1.0000    1.0000    1.0000        60
      scratches     0.9516    0.9833    0.9672        60

       accuracy                         0.9889       360
```

4 classes sur 6 atteignent un F1 de 1.0. Les légères confusions sur `inclusion` et `scratches` s'expliquent par la ressemblance visuelle entre ces défauts.


## Analyse ROI

Le module `roi.py` simule un contexte industriel réaliste (ligne de production acier, 120 pièces/heure, 5% de taux de défaut) pour estimer le retour sur investissement par rapport à une inspection humaine à 72% de précision.

Hypothèses utilisées :
- Coût d'un défaut non détecté : **150 €**
- Coût d'une fausse alarme : **12 €**
- 2 inspecteurs remplacés / assistés à 38 000 €/an chacun
- Coût de déploiement : 35 000 € (matériel + intégration)
- Maintenance annuelle : 4 000 €/an

Le graphe `roi_analysis.png` montre le ROI cumulé sur 36 mois, la comparaison des coûts annuels, et la sensibilité du ROI à la précision du modèle.


## Installation et lancement

```bash
# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'entraînement complet 
python main.py

# Démo rapide sur le modèle sauvegardé
python demo.py
```

Le dossier `outputs/` est créé automatiquement. Tous les graphes et rapports y sont sauvegardés.

---

## Dépendances

- Python 3.10+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib / seaborn
- NumPy / Pillow

*KODJO Messanh Yaovi — Deep Learning Appliqué en Entreprise*
