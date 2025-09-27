# Microscopy Cell Counting (DL)

PyTorch project to **count cells** in 256×256 microscopy images by regressing the total count from an image.
Uses a small fully‑convolutional CNN on the **blue channel** with light flips, MSE loss, Adam, and MAE as the key metric.

## Data
Paired folders of 256×256 microscopy RGB images with cells of varying sizes, and binary “dots” labels with one dot at each cell center (same size, 256×256). Example layout:
```
counting_cells_data/
  train_images/    # input images (e.g., ...123cell.png)
  train_labels/    # labels (e.g., 123dots.png)
  val_images/
  val_labels/
```
> Files are matched by the 3 digits before “cell” → `{XXX}dots.png`.

## Quickstart
1) **Install**
```
pip install torch torchvision matplotlib numpy pillow
```
2) **Set dataset paths** in `model_training_nb.ipynb`:
```
train_folder       = ".../counting_cells_data/train_images"
train_labels_folder= ".../counting_cells_data/train_labels"
val_folder         = ".../counting_cells_data/val_images"
val_label_folder   = ".../counting_cells_data/val_labels"
```
3) **Run the notebook** (CPU or GPU). Artifacts are saved to:
```
./results_blue_unnorm_flips/
  ├── best_model_blue_unnorm_flips.pth
  └── training_curves_blue_unnorm_flips.png
```

## Notes
- Model: `SimpleConvCounter(input_channels=1, initial_filters=32)`; **blue‑channel only** (`select_blue_channel`), **no normalization**.
- Default training config: `batch_size=32`, `epochs=10`, `lr=1e-3`, `MSELoss → minimize MAE`.
- Reproducibility: fixed seeds; deterministic dataloaders on Windows (`num_workers=0`).

---
Data from "Detecting Repeating Objects using Patch Correlation Analysis" by Inbar Huberman-Spiegelglas

