# Métriques d'évaluation pour la segmentation (sémantique et d'instance)

Devoir du Master Data & IA : *Evaluation Metrics for Computer Vision Tasks*.
Ce dépôt contient le code de la partie **segmentation** : calcul des métriques sur des masques synthétiques (aucun dataset à télécharger).

## Contenu

| Fichier | Rôle |
|---|---|
| `segmentation_metrics.py` | Script principal : métriques, courbes, démonstration |
| `segmentation_metrics.ipynb` | Même code, présenté étape par étape avec explications |
| `confusion_matrix.png`, `roc_pr.png` | Figures générées par le script |
| `segmentation_metriques.pptx` | Présentation |

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

## Utilisation

```bash
python segmentation_metrics.py
```

Le script affiche la matrice de confusion, les métriques par classe, l'AP d'instance et les métriques de régression, puis enregistre `confusion_matrix.png` et `roc_pr.png`.

## Documentation du code

### Données
- `make_data(h, w)` : crée une vérité terrain à 3 classes (0 = fond, 1 = route, 2 = voiture) et une prédiction obtenue par décalage et bruit aléatoire. Retourne `(gt, pred)`.

### Segmentation sémantique
- `confusion_matrix(gt, pred, n)` : matrice `n × n` (lignes = classe réelle, colonnes = classe prédite) calculée avec `np.bincount`.
- `per_class_metrics(cm)` : retourne un dictionnaire avec
  - `precision` = TP / (TP + FP)
  - `recall` = TP / (TP + FN)
  - `f1_dice` = 2TP / (2TP + FP + FN)
  - `iou` = TP / (TP + FP + FN), et `miou` (moyenne sur les classes)
  - `pixel_acc` = pixels corrects / pixels totaux
- `plot_confusion(cm)` : enregistre la matrice normalisée par ligne dans `confusion_matrix.png`.

### Segmentation d'instance
- `mask_iou(a, b)` : IoU entre deux masques binaires.
- `average_precision(gt_masks, pred_masks, scores, iou_thr)` : AP à un seuil d'IoU. Les prédictions sont triées par score décroissant, appariées une à une à un masque réel non encore utilisé, puis l'aire sous la courbe précision-rappel est intégrée.
- `instance_demo()` : exemple avec 2 objets réels et 3 prédictions. Résultat : AP@0,5 = 1,0 et AP@0,75 = 0,5.

### Courbes pixel par pixel
- `roc_pr_demo(gt)` : simule des probabilités pour la classe voiture, trace la courbe ROC (avec AUC) et la courbe précision-rappel (avec AP), enregistre `roc_pr.png`.

### Contre-exemple
- `regression_metrics_demo(y_true, y_score)` : MSE, RMSE et R² sur les mêmes données. Ces métriques sont des métriques de **régression** : elles ne sont pas adaptées à la classification de pixels (R² négatif dans la démo).

## Résultats de la démo

| Métrique | Valeur |
|---|---|
| mIoU | 0,70 |
| Pixel accuracy | 0,88 |
| AP@0,5 / AP@0,75 (instance) | 1,0 / 0,5 |
| MSE / RMSE / R² | 0,077 / 0,278 / −0,35 |

## Pour aller plus loin

Remplacer les masques synthétiques par un vrai modèle (U-Net, DeepLab, Mask R-CNN de `torchvision`) et utiliser `pycocotools` pour le mAP COCO (seuils 0,5 à 0,95) et la Panoptic Quality.
