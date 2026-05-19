# GTSRB — Reconnaissance de Panneaux de Signalisation

Projet de Deep Learning pour la classification automatique de panneaux de signalisation routière sur le dataset **GTSRB** (German Traffic Sign Recognition Benchmark).

---

## Dataset

| Propriété | Valeur |
|-----------|--------|
| Classes | 43 types de panneaux |
| Images d'entraînement | ~39 000 |
| Images de test | ~12 600 |
| Taille d'entrée | 64×64 px |
| Source | [GTSRB sur Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) |

---

## Modèles entraînés

### CNN from scratch — `gtsrb_cnn_final.keras`
- 3 blocs Conv2D + BatchNormalization + MaxPooling (32 → 64 → 128 filtres)
- Tête de classification : Dense 512 + Dense 256 + Dropout
- Optimizer : Adam | Callbacks : EarlyStopping, ReduceLROnPlateau
- **Accuracy test : ~85%** | ~2M paramètres

### Transfer Learning (EfficientNetB0) — `gtsrb_efficientnet_final.keras`
- Base : EfficientNetB0 pré-entraîné sur ImageNet
- Tête personnalisée : GlobalAveragePooling2D + Dense 256 + Dense 128
- Fine-tuning des 30 dernières couches (LR = 1e-5)
- **Accuracy test : ~97%** | ~5.3M paramètres

---

## Résultats

| Modèle | Accuracy test |
|--------|--------------|
| CNN from scratch | ~85% |
| EfficientNetB0 (Transfer Learning) | **~97%** |

---

## Prétraitement & Augmentation

- Resize → 64×64 px, normalisation [0, 1]
- Augmentation : rotation ±15°, zoom ±15%, décalage ±10%, luminosité [0.8–1.2]
- **Pas de flip horizontal** — les panneaux sont asymétriques (ex. sens interdit ≠ sens autorisé)
- EarlyStopping + ReduceLROnPlateau pour éviter le surapprentissage

---

## Lancer l'application Streamlit

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'app
streamlit run streamlit_app.py
```

L'application s'ouvre sur `http://localhost:8501` et propose :
- **Prédiction en temps réel** : uploadez une image (JPG, PNG, PPM, BMP), obtenez le top-5 des prédictions avec score de confiance
- **Comparaison des modèles** : métriques, architectures CNN vs EfficientNetB0, stratégie d'entraînement
- **Explorateur de classes** : les 43 panneaux organisés par catégorie, avec filtre de recherche

---

## Structure du projet

```
GTSRB/
├── GTSRB_Deep_Learning.ipynb       # Notebook d'entraînement (Kaggle)
├── streamlit_app.py                 # Application de démonstration
├── gtsrb_efficientnet_final.keras  # Modèle EfficientNetB0 (~97%)
├── gtsrb_cnn_final.keras           # Modèle CNN scratch (~85%)
├── class_indices.json              # Mapping index → classe
├── class_names.json                # Noms des 43 classes (FR)
├── requirements.txt
└── README.md
```

---

## Points clés

- **EfficientNetB0** : gain de +12 points d'accuracy grâce au pré-entraînement ImageNet
- **Fine-tuning en 2 phases** : tête seule (LR = 1e-3), puis 30 dernières couches (LR = 1e-5)
- **Dataset déséquilibré** : certaines classes ont 10× plus d'images que d'autres — l'augmentation compense
- **Entraîné sur Kaggle** avec GPU T4 x2

---

## Références

- [GTSRB Benchmark](http://benchmark.ini.rub.de/?section=gtsrb)
- [EfficientNet paper — Tan & Le, 2019](https://arxiv.org/abs/1905.11946)

*Projet réalisé sur Kaggle Notebooks — Paris School of Technology & Business*
