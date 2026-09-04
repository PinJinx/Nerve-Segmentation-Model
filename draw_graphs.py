"""
draw_graphs.py
==============
Model performance visualizer and statistical analyzer for nerve segmentation models.

Features:
1. Displays Model Specification (Backbone, Decoder, Parameters, Inputs/Outputs).
2. Generates side-by-side prediction samples:
   - 5 images with prediction (nerve detected)
   - 5 images with no prediction (no nerve detected)
   Composite panels: Raw Image | Predicted Mask Heatmap | Overlay with nerve probability badge.
3. Produces a comprehensive statistical performance report:
   - Key Metrics Table (F1 Score, Accuracy, Precision, Recall, Specificity, IoU, Dice, ROC-AUC, PR-AUC)
   - Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives)
   - ROC & Precision-Recall Curves with AUC scores
   - F1 / Dice Score vs Decision Threshold sweep
   - Segmentation Dice & IoU Score Distributions
   - Predicted Nerve Probability & Pixel Coverage Distributions

Usage:
    python draw_graphs.py --model resnet34_plus
    python draw_graphs.py --model unet_inception
    python draw_graphs.py --model se_resnext50

All output saved to <project>/result/ (created automatically).
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from data import load_test_ids, load_test, load_train, load_pickle, save_pickle
from train import RES_DIR, get_model_paths, resize_batch, split_random, nerve_presence, SegDataset, DEVICE
from model import IMG_ROWS, IMG_COLS, MODEL_REGISTRY, MODEL_ALIASES, get_model, get_model_spec, resolve_model_name

# ── Output directory ───────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR  = os.path.join(PROJECT_DIR, 'result')

# ── Colour palette ─────────────────────────────────────────────────────────────
ACCENT  = "#5E81F4"
ACCENT2 = "#56CFE1"
WARN    = "#FF6B6B"
GREEN   = "#57CC99"
PURPLE  = "#B5179E"
BG      = "#0F1117"
CARD    = "#1A1D27"
CARD2   = "#202436"
TEXT    = "#E8EAF6"
SUBTEXT = "#A0A5C0"
GRID_C  = "#2A2D3E"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   GRID_C,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID_C,
    "grid.linestyle":   "--",
    "grid.alpha":       0.45,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  2,
})

THRESHOLD = 0.5


# ── Load Predictions & Ground Truth ───────────────────────────────────────────

def load_test_predictions(model_name='resnet34_plus'):
    canonical = resolve_model_name(model_name)
    weights_path, history_path, val_eval_path, test_masks_path, test_probs_path = get_model_paths(canonical)
    
    if not os.path.exists(test_masks_path) or not os.path.exists(test_probs_path):
        raise FileNotFoundError(
            f"Prediction files not found for [{canonical}]: '{test_masks_path}'. "
            f"Please run 'python predict.py --model {canonical}' first."
        )

    ids   = load_test_ids()
    masks = np.load(test_masks_path)   # (N, 1, H, W) float32 0-1
    probs = np.load(test_probs_path)   # (N,) float32 0-1
    order = np.argsort(ids)
    ids, masks, probs = ids[order], masks[order], probs[order]

    print(f"Loading test images for model [{canonical}]...")
    imgs = load_test()
    imgs = resize_batch(imgs)          # (N, 1, H, W) uint8
    imgs = imgs[order]
    return ids, imgs, masks, probs


def load_or_generate_val_evaluation(model_name='resnet34_plus'):
    """Load validation evaluation data (with ground truth) or generate if weights exist."""
    canonical = resolve_model_name(model_name)
    weights_path, history_path, val_eval_path, test_masks_path, test_probs_path = get_model_paths(canonical)

    if os.path.exists(val_eval_path):
        print(f"Loading validation ground truth & evaluation payload for [{canonical}]...")
        return load_pickle(val_eval_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No trained weights found for [{canonical}] at '{weights_path}'. "
            f"Please run 'python train.py --model {canonical}' first."
        )

    print(f"Validation evaluation payload not found. Evaluating [{canonical}] on validation set...")
    imgs, masks = load_train()
    imgs  = resize_batch(imgs)
    masks = resize_batch(masks)
    
    x_tr, y_tr, x_val, y_val = split_random(imgs, masks, val_split=0.2)
    mean, std = x_tr.astype(np.float32).mean(), x_tr.astype(np.float32).std()
    y_val_f = y_val.astype(np.float32) / 255.0

    val_ds = SegDataset(x_val, y_val_f, mean, std)
    from torch.utils.data import DataLoader
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = get_model(canonical).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    all_val_segs, all_val_auxs = [], []
    import torch
    with torch.no_grad():
        for val_imgs_b, _, _ in val_dl:
            val_imgs_b = val_imgs_b.to(DEVICE)
            seg_b, aux_b = model(val_imgs_b)
            all_val_segs.append(seg_b.cpu().numpy())
            all_val_auxs.append(aux_b.reshape(-1).cpu().numpy())

    val_seg_preds = np.concatenate(all_val_segs, axis=0)
    val_aux_preds = np.concatenate(all_val_auxs, axis=0)

    val_eval_data = {
        'x_val': x_val,
        'y_val_true': y_val_f,
        'y_val_nerve': nerve_presence(y_val_f),
        'y_val_seg_pred': val_seg_preds,
        'y_val_aux_pred': val_aux_preds,
    }
    save_pickle(val_eval_path, val_eval_data)
    return val_eval_data


# ── Sample Composite Generation ───────────────────────────────────────────────

def _composite(img_hw, mask_hw, prob):
    """Return an RGB composite: raw image | mask heatmap | overlay."""
    H, W = img_hw.shape
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)

    # panel 0 – raw grayscale
    gray3 = cv2.cvtColor(img_hw, cv2.COLOR_GRAY2BGR)
    canvas[:, :W] = gray3

    # panel 1 – plasma mask heatmap
    mask_uint8 = (mask_hw * 255).clip(0, 255).astype(np.uint8)
    plasma = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_PLASMA)
    canvas[:, W:2 * W] = plasma

    # panel 2 – overlay
    heat  = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    alpha = mask_hw[..., None]
    overlay = (gray3 * (1 - alpha * 0.6) + heat * alpha * 0.6).clip(0, 255).astype(np.uint8)
    canvas[:, 2 * W:] = overlay

    # dividers
    canvas[:, W - 1:W + 1]     = 60
    canvas[:, 2 * W - 1:2 * W + 1] = 60

    # header bar
    bar_h = 22
    header = np.zeros((bar_h, W * 3, 3), dtype=np.uint8)
    header[:] = (26, 29, 39)
    for i, lbl in enumerate(["Raw Image", "Predicted Mask", "Overlay"]):
        cv2.putText(header, lbl, (i * W + 6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    prob_str = f"nerve p={prob:.3f}"
    cv2.putText(header, prob_str, (W * 3 - 120, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (86, 201, 255) if prob >= THRESHOLD else (100, 107, 255), 1, cv2.LINE_AA)

    return np.vstack([header, canvas])


def save_samples(ids, imgs, masks, probs, model_name='resnet34_plus', n=5):
    """Save n nerve + n no-nerve composite PNGs to result/samples_<model>/."""
    canonical = resolve_model_name(model_name)
    nerve_idx    = np.where(probs >= THRESHOLD)[0]
    no_nerve_idx = np.where(probs < THRESHOLD)[0]

    rng = np.random.default_rng(42)
    nerve_sel    = rng.choice(nerve_idx,    size=min(n, len(nerve_idx)),    replace=False) if len(nerve_idx) > 0 else []
    no_nerve_sel = rng.choice(no_nerve_idx, size=min(n, len(no_nerve_idx)), replace=False) if len(no_nerve_idx) > 0 else []

    out_base = os.path.join(RESULT_DIR, f"samples_{canonical}")
    for subfolder, indices, tag in [
        ("with_prediction", nerve_sel,    "nerve"),
        ("no_prediction",   no_nerve_sel, "no_nerve"),
    ]:
        out_dir = os.path.join(out_base, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        for rank, idx in enumerate(indices, 1):
            comp = _composite(imgs[idx, 0], masks[idx, 0], probs[idx])
            fname = f"{tag}_{rank:02d}_id{ids[idx]}.png"
            cv2.imwrite(os.path.join(out_dir, fname), comp)

    print(f"  Saved prediction sample composites -> {out_base}/")


# ── Statistical Metrics Computation ──────────────────────────────────────────

def compute_detailed_metrics(val_eval_data, thresh=0.5):
    y_true_mask  = val_eval_data['y_val_true']      # (N, 1, H, W)
    y_true_nerve = val_eval_data['y_val_nerve']     # (N,)
    y_pred_mask  = val_eval_data['y_val_seg_pred']  # (N, 1, H, W)
    y_pred_prob  = val_eval_data['y_val_aux_pred']  # (N,)

    N = len(y_true_nerve)

    # 1. Classification Metrics (Nerve Existence Aux Head)
    y_pred_nerve = (y_pred_prob >= thresh).astype(np.float32)
    tp = int(((y_pred_nerve == 1) & (y_true_nerve == 1)).sum())
    fp = int(((y_pred_nerve == 1) & (y_true_nerve == 0)).sum())
    tn = int(((y_pred_nerve == 0) & (y_true_nerve == 0)).sum())
    fn = int(((y_pred_nerve == 0) & (y_true_nerve == 1)).sum())

    cls_acc   = (tp + tn) / (N + 1e-7)
    cls_prec  = tp / (tp + fp + 1e-7)
    cls_rec   = tp / (tp + fn + 1e-7)
    cls_spec  = tn / (tn + fp + 1e-7)
    cls_f1    = 2 * cls_prec * cls_rec / (cls_prec + cls_rec + 1e-7)

    # 2. Segmentation Mask Metrics (Dice & IoU)
    pred_bin_mask = (y_pred_mask >= thresh).astype(np.float32)
    
    # Per-sample Dice & IoU
    intersection = (pred_bin_mask * y_true_mask).sum(axis=(1, 2, 3))
    total_pixels = pred_bin_mask.sum(axis=(1, 2, 3)) + y_true_mask.sum(axis=(1, 2, 3))
    union        = total_pixels - intersection

    sample_dice = (2.0 * intersection + 1e-6) / (total_pixels + 1e-6)
    sample_iou  = (intersection + 1e-6) / (union + 1e-6)

    mean_dice   = sample_dice.mean()
    median_dice = np.median(sample_dice)
    mean_iou    = sample_iou.mean()
    median_iou  = np.median(sample_iou)

    # Pixel Accuracy
    correct_px = (pred_bin_mask == y_true_mask).sum()
    total_px   = pred_bin_mask.size
    px_acc     = correct_px / total_px

    # 3. ROC & PR Curve Computation
    thresholds = np.linspace(0.0, 1.0, 101)
    tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
    f1_sweep = []

    for t in thresholds:
        yp = (y_pred_prob >= t).astype(int)
        tp_t = ((yp == 1) & (y_true_nerve == 1)).sum()
        fp_t = ((yp == 1) & (y_true_nerve == 0)).sum()
        tn_t = ((yp == 0) & (y_true_nerve == 0)).sum()
        fn_t = ((yp == 0) & (y_true_nerve == 1)).sum()

        tpr = tp_t / (tp_t + fn_t + 1e-7)
        fpr = fp_t / (fp_t + tn_t + 1e-7)
        prec = tp_t / (tp_t + fp_t + 1e-7)
        rec = tpr
        f1 = 2 * prec * rec / (prec + rec + 1e-7)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_sweep.append(f1)

    tpr_arr  = np.array(tpr_list)
    fpr_arr  = np.array(fpr_list)
    prec_arr = np.array(prec_list)
    rec_arr  = np.array(rec_list)
    f1_arr   = np.array(f1_sweep)

    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    roc_order = np.argsort(fpr_arr)
    roc_auc   = float(trapz_fn(tpr_arr[roc_order], fpr_arr[roc_order]))

    pr_order  = np.argsort(rec_arr)
    pr_auc    = float(trapz_fn(prec_arr[pr_order], rec_arr[pr_order]))

    best_thresh_idx = np.argmax(f1_arr)
    opt_thresh      = thresholds[best_thresh_idx]
    opt_f1          = f1_arr[best_thresh_idx]

    return {
        "N": N,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "cls_acc": cls_acc, "cls_prec": cls_prec, "cls_rec": cls_rec,
        "cls_spec": cls_spec, "cls_f1": cls_f1,
        "mean_dice": mean_dice, "median_dice": median_dice,
        "mean_iou": mean_iou, "median_iou": median_iou,
        "px_acc": px_acc,
        "sample_dice": sample_dice, "sample_iou": sample_iou,
        "thresholds": thresholds, "f1_sweep": f1_arr,
        "tpr_arr": tpr_arr, "fpr_arr": fpr_arr,
        "prec_arr": prec_arr, "rec_arr": rec_arr,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "opt_thresh": opt_thresh, "opt_f1": opt_f1,
        "y_true_nerve": y_true_nerve, "y_pred_prob": y_pred_prob,
        "y_pred_mask": y_pred_mask,
    }


# ── KDE Helper ────────────────────────────────────────────────────────────────

def _plot_kde(ax, data, color, label=None):
    data = data[np.isfinite(data)]
    if len(data) < 2 or data.std() < 1e-9:
        return
    kde = gaussian_kde(data, bw_method=0.25)
    xs  = np.linspace(data.min(), data.max(), 300)
    ax2 = ax.twinx()
    ax2.plot(xs, kde(xs), color=color, lw=2, label=label)
    ax2.set_ylabel("Density", color=color, fontsize=8)
    ax2.tick_params(axis="y", colors=color)
    ax2.spines["right"].set_color(color)
    ax2.set_facecolor("none")


# ── Comprehensive Performance Report Figure ───────────────────────────────────

def plot_model_performance(val_eval_data, model_name='resnet34_plus'):
    canonical = resolve_model_name(model_name)
    spec = get_model_spec(canonical)
    metrics = compute_detailed_metrics(val_eval_data, thresh=THRESHOLD)

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        f"Statistical Performance Report & Analysis [{spec['display_name']}]",
        fontsize=16, color=TEXT, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    # # ── 1. Model Specification Card & Key Metrics Table (Top-Left)
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.axis("off")
    # ax0.set_title("Model Specification & Metrics Summary", fontsize=11, pad=10, fontweight="bold")

    # table_data = [
    #     ["Model Name",       f"{spec['display_name']} ({spec['canonical_name']})"],
    #     ["Encoder Backbone", spec['backbone']],
    #     ["Decoder Type",     spec['decoder']],
    #     ["Total Parameters", f"{spec['total_params_fmt']} ({spec['total_params']:,})"],
    #     ["Val Samples (N)",  f"{metrics['N']}"],
    #     ["Accuracy (Cls)",   f"{metrics['cls_acc'] * 100:.2f}%"],
    #     ["F1 Score (Cls)",   f"{metrics['cls_f1']:.4f}"],
    #     ["Precision / Rec",  f"{metrics['cls_prec']:.4f} / {metrics['cls_rec']:.4f}"],
    #     ["Specificity",      f"{metrics['cls_spec']:.4f}"],
    #     ["ROC-AUC / PR-AUC", f"{metrics['roc_auc']:.4f} / {metrics['pr_auc']:.4f}"],
    #     ["Mean Dice (Mask)", f"{metrics['mean_dice']:.4f} (Med: {metrics['median_dice']:.4f})"],
    #     ["Mean IoU (Mask)",  f"{metrics['mean_iou']:.4f} (Med: {metrics['median_iou']:.4f})"],
    #     ["Optimal Threshold",f"{metrics['opt_thresh']:.2f} (Peak F1: {metrics['opt_f1']:.4f})"],
    # ]

    # table = ax0.table(
    #     cellText=table_data,
    #     colLabels=["Specification / Metric", "Value / Score"],
    #     cellLoc="left",
    #     loc="center",
    #     bbox=[0, -0.05, 1, 1.05],
    # )
    # table.auto_set_font_size(False)
    # table.set_fontsize(8.5)
    # for (r, c), cell in table.get_celld().items():
    #     cell.set_edgecolor(GRID_C)
    #     if r == 0:
    #         cell.set_facecolor("#151824")
    #         cell.set_text_props(color=ACCENT2, fontweight="bold")
    #     else:
    #         cell.set_facecolor(CARD if r % 2 == 0 else CARD2)
    #         cell.set_text_props(color=TEXT)
    #         if c == 1 and ("F1" in table_data[r - 1][0] or "Dice" in table_data[r - 1][0] or "Accuracy" in table_data[r - 1][0]):
    #             cell.set_text_props(color=GREEN, fontweight="bold")

    # ── 2. Confusion Matrix Heatmap (Top-Middle)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("Nerve Existence Confusion Matrix", fontsize=11, fontweight="bold")
    cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    im = ax1.imshow(cm, cmap="Blues", interpolation="nearest")
    
    labels = [["TN\n(True Neg)", "FP\n(False Pos)"], ["FN\n(False Neg)", "TP\n(True Pos)"]]
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / metrics['N'] * 100
            txt_color = "white" if cm[i, j] > cm.max() / 2 else TEXT
            ax1.text(j, i, f"{labels[i][j]}\n{count}\n({pct:.1f}%)",
                     ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Predicted No-Nerve", "Predicted Nerve"])
    ax1.set_yticklabels(["Actual No-Nerve", "Actual Nerve"])
    ax1.grid(False)

    # ── 3. ROC & Precision-Recall Curves (Top-Right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("ROC & Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax2.plot(metrics['fpr_arr'], metrics['tpr_arr'], color=ACCENT, lw=2.5,
             label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax2.plot(metrics['rec_arr'], metrics['prec_arr'], color=ACCENT2, lw=2.5, ls="--",
             label=f"PR  (AUC = {metrics['pr_auc']:.3f})")
    ax2.plot([0, 1], [0, 1], color=GRID_C, lw=1.2, ls=":", label="Random Guess")
    ax2.set_xlabel("False Positive Rate / Recall")
    ax2.set_ylabel("True Positive Rate / Precision")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True)

    # ── 4. F1 Score vs Threshold Sweep Curve (Bottom-Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("F1 Score vs Decision Threshold Sweep", fontsize=11, fontweight="bold")
    ax3.plot(metrics['thresholds'], metrics['f1_sweep'], color=GREEN, lw=2.5, label="F1 Score")
    ax3.axvline(metrics['opt_thresh'], color=WARN, lw=1.5, ls="--",
                label=f"Optimal Thresh {metrics['opt_thresh']:.2f} (F1={metrics['opt_f1']:.3f})")
    ax3.axvline(0.5, color="white", lw=1.0, ls=":", label="Default 0.50")
    ax3.set_xlabel("Probability Threshold")
    ax3.set_ylabel("Classification F1 Score")
    ax3.legend(fontsize=8.5, loc="lower center")
    ax3.grid(True)

    # ── 5. Segmentation Dice & IoU Distribution (Bottom-Middle)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Segmentation Mask Dice & IoU Distributions", fontsize=11, fontweight="bold")
    ax4.hist(metrics['sample_dice'], bins=40, color=ACCENT, alpha=0.6, label="Dice Score", edgecolor=BG)
    ax4.hist(metrics['sample_iou'], bins=40, color=PURPLE, alpha=0.5, label="IoU Score", edgecolor=BG)
    ax4.axvline(metrics['mean_dice'], color=ACCENT, lw=1.5, ls="--", label=f"Mean Dice ({metrics['mean_dice']:.3f})")
    ax4.axvline(metrics['mean_iou'], color=PURPLE, lw=1.5, ls=":", label=f"Mean IoU ({metrics['mean_iou']:.3f})")
    _plot_kde(ax4, metrics['sample_dice'], ACCENT2)
    ax4.set_xlabel("Score Value")
    ax4.set_ylabel("Sample Count")
    ax4.legend(fontsize=8, loc="upper center")
    ax4.grid(True)

    # ── 6. Predicted Probability Distribution (Bottom-Right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Predicted Nerve Probability Distribution", fontsize=11, fontweight="bold")
    nerve_pos = metrics['y_pred_prob'][metrics['y_true_nerve'] == 1]
    nerve_neg = metrics['y_pred_prob'][metrics['y_true_nerve'] == 0]

    ax5.hist(nerve_pos, bins=30, color=GREEN, alpha=0.65, label=f"Actual Nerve (N={len(nerve_pos)})", edgecolor=BG)
    ax5.hist(nerve_neg, bins=30, color=WARN, alpha=0.55, label=f"Actual No-Nerve (N={len(nerve_neg)})", edgecolor=BG)
    ax5.axvline(THRESHOLD, color="white", lw=1.5, ls="--", label=f"Threshold {THRESHOLD}")
    ax5.set_xlabel("Auxiliary Head Predicted Probability")
    ax5.set_ylabel("Sample Count")
    ax5.legend(fontsize=8.5, loc="upper center")
    ax5.grid(True)

    plt.tight_layout()
    out_file = os.path.join(RESULT_DIR, f"{canonical}_performance_report.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved performance report -> {out_file}")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    valid_choices = sorted(list(set(list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys()))))
    parser = argparse.ArgumentParser(description="Generate model performance report and prediction samples")
    parser.add_argument('--model', type=str, default='resnet34_plus', choices=valid_choices,
                        help='Model architecture to evaluate and plot (resnet34_plus, unet_inception, se_resnext50, etc.)')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample composite images to save per class')
    args = parser.parse_args()

    canonical = resolve_model_name(args.model)
    os.makedirs(RESULT_DIR, exist_ok=True)

    spec = get_model_spec(canonical)
    print("\n" + "=" * 60)
    print(f" GENERATING PERFORMANCE REPORT FOR MODEL: {spec['display_name']}")
    print(f" Canonical Key: {spec['canonical_name']}")
    print(f" Backbone:      {spec['backbone']}")
    print(f" Decoder:       {spec['decoder']}")
    print(f" Parameters:    {spec['total_params_fmt']} ({spec['total_params']:,})")
    print(f" Inputs/Outputs:{spec['input_shape']} -> {spec['outputs']}")
    print("=" * 60 + "\n")

    # 1. Prediction Samples
    print("[1/2] Generating prediction sample composites (raw | mask | overlay)...")
    try:
        ids, imgs, masks, probs = load_test_predictions(canonical)
        save_samples(ids, imgs, masks, probs, model_name=canonical, n=args.samples)
    except FileNotFoundError as e:
        print(f"  NOTICE: Skipping test sample composites. Reason: {e}")

    # 2. Performance Report Graphs
    print("\n[2/2] Generating statistical performance graphs & metrics report...")
    val_eval_data = load_or_generate_val_evaluation(canonical)
    plot_model_performance(val_eval_data, model_name=canonical)

    print(f"\nCompleted analysis for [{canonical}]. Outputs saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()

