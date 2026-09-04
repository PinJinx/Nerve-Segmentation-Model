# Ultrasound Nerve Segmentation — Modern PyTorch

Modernised rewrite of [EdwardTyantov/ultrasound-nerve-segmentation](https://github.com/EdwardTyantov/ultrasound-nerve-segmentation).

**Stack:** Python 3 · PyTorch 2 · OpenCV

---

## Files

| File | Purpose |
|---|---|
| `data.py` | Converts raw `.tif` images to `.npy` arrays, loads them |
| `model.py` | U-Net with inception blocks, residual skip connections, dual-head output |
| `train.py` | Training loop, augmentation, early stopping, saves weights |
| `predict.py` | Inference with TTA + multi-model ensemble support |
| `submission.py` | Generates Kaggle `submission.csv` |

---

## Setup

```bash
pip install -r requirements.txt
```

Place competition data at `../../train/` and `../../test/` (relative to this folder), then:

```bash
python data.py
```

---

## Train

```bash
# Random split (default)
python train.py

# Patient-based split
python train.py --split patient

# All options
python train.py --split random --epochs 50 --batch 64 --lr 0.0045 --patience 5
```

Weights are saved to `res/unet.pt`.

---

## Predict

Single model with test-time augmentation (TTA):

```bash
python predict.py
```

K-fold ensemble:

```bash
python predict.py --kfold_paths res/fold0.pt res/fold1.pt res/fold2.pt
```

---

## Submit

```bash
python submission.py
```

Output: `res/submission.csv`

---

## Architecture

- **Encoder**: 4× (InceptionBlock → strided Conv2d downsample → Dropout)
- **Bottleneck**: InceptionBlock + Dropout
- **Auxiliary head**: nerve-presence classifier (binary cross-entropy, weight 0.5)
- **Decoder**: 4× (bilinear upsample → residual skip → InceptionBlock → Dropout)
- **Output**: per-pixel sigmoid segmentation mask
- **Loss**: Dice loss (segmentation) + BCE (nerve presence)
- **Optimizer**: Adam, LR decayed by `ReduceLROnPlateau`

Each InceptionBlock has 4 parallel branches (1×1, factorised 3×3, factorised 5×5, pooling).
