# Microscopy Cell Counting (PyTorch)

A compact CNN regressor for microscopy **cell counting** using label masks as supervision (count = non‑zero pixels). Notebook-first repo; lightweight and reproducible.

## TL;DR
- **Data**: 200 images (train/val split ~180/20). Single-channel (blue) pipeline; H/V flips.
- **Model**: 4×(Conv‑BN‑ReLU‑MaxPool) → GlobalAvgPool → Linear (count). Loss: **MSE**. Optim: **Adam**. **StepLR** scheduler.
- **Training**: ~130 epochs, fixed seed, best‑epoch checkpointing.
- **Result**: **Validation MAE ↓ ~170 → ~2.69** (notebook logs at end).

## Repo Layout
```
.
├─ notebooks/
│  └─ counting_cells.ipynb        # main notebook
├─ data/
│  └─ sample/                     # tiny sample for structure only
│     ├─ train_images/  train_labels/
│     └─ val_images/    val_labels/
├─ requirements.txt
└─ README.md
```

## Data
- Labels are binary masks; the *target count* is the number of non‑zero pixels in the label image.
- This repo ships **only a tiny sample** under `data/sample/` for structure. Use the full dataset locally.
- Update paths in the notebook’s first cell if needed.

## Results (Table)
| Split | MAE | Notes |
|------:|----:|:------|
| Val   | **2.69** | Best‑epoch checkpoint; ~130 epochs |
