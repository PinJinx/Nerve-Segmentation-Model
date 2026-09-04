import os
import shutil
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import load_train, load_test, load_patient_nums, save_pickle
from model import (
    get_model,
    get_model_spec,
    resolve_model_name,
    MODEL_REGISTRY,
    MODEL_ALIASES,
    dice_loss,
    dice_coef,
    IMG_ROWS,
    IMG_COLS,
)

RES_DIR      = os.path.join(os.path.dirname(__file__), 'res')
MEANSTD_PATH = os.path.join(RES_DIR, 'meanstd.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_paths(model_name: str):
    canonical       = resolve_model_name(model_name)
    weights_path    = os.path.join(RES_DIR, f'{canonical}_weights.pt')
    history_path    = os.path.join(RES_DIR, f'{canonical}_history.pkl')
    val_eval_path   = os.path.join(RES_DIR, f'{canonical}_val_eval.pkl')
    test_masks_path = os.path.join(RES_DIR, f'{canonical}_imgs_mask_test.npy')
    test_probs_path = os.path.join(RES_DIR, f'{canonical}_imgs_mask_exist_test.npy')

    # Fallbacks for legacy files
    if not os.path.exists(weights_path) and canonical == 'unet_inception' and os.path.exists(os.path.join(RES_DIR, 'unet.pt')):
        weights_path = os.path.join(RES_DIR, 'unet.pt')

    if not os.path.exists(history_path) and os.path.exists(os.path.join(RES_DIR, 'train_history.pkl')):
        history_path = os.path.join(RES_DIR, 'train_history.pkl')

    if not os.path.exists(test_masks_path) and os.path.exists(os.path.join(RES_DIR, 'imgs_mask_test.npy')):
        test_masks_path = os.path.join(RES_DIR, 'imgs_mask_test.npy')

    if not os.path.exists(test_probs_path) and os.path.exists(os.path.join(RES_DIR, 'imgs_mask_exist_test.npy')):
        test_probs_path = os.path.join(RES_DIR, 'imgs_mask_exist_test.npy')

    return weights_path, history_path, val_eval_path, test_masks_path, test_probs_path


def resize_batch(imgs):
    out = np.zeros((imgs.shape[0], 1, IMG_ROWS, IMG_COLS), dtype=np.uint8)
    for i, img in enumerate(imgs):
        out[i, 0] = cv2.resize(img[0], (IMG_COLS, IMG_ROWS), interpolation=cv2.INTER_CUBIC)
    return out


def nerve_presence(masks):
    return np.array([1.0 if masks[i, 0].sum() > 0 else 0.0 for i in range(len(masks))], dtype=np.float32)


def augment(imgs, masks):
    xs, ys = [], []
    for img, mask in zip(imgs, masks):
        xs.append(img);  ys.append(mask)
        xs.append(img[:, :, ::-1].copy()); ys.append(mask[:, :, ::-1].copy())
        xs.append(img[:, ::-1, :].copy()); ys.append(mask[:, ::-1, :].copy())
        zx, zy = np.random.uniform(0.9, 1.1, 2)
        M = cv2.getRotationMatrix2D((IMG_COLS / 2, IMG_ROWS / 2), 0, 1.0)
        zoom_m = np.float32([[zx, 0, (1 - zx) * IMG_COLS / 2], [0, zy, (1 - zy) * IMG_ROWS / 2]])
        xs.append(cv2.warpAffine(img[0], zoom_m, (IMG_COLS, IMG_ROWS))[None])
        ys.append(cv2.warpAffine(mask[0], zoom_m, (IMG_COLS, IMG_ROWS))[None])
        shift = np.random.uniform(-5.0, 5.0)
        xs.append(np.clip(img + shift, 0, None)); ys.append(mask)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def split_random(imgs, masks, val_split=0.2):
    n = len(imgs)
    idx = np.random.permutation(n)
    cut = int(n * (1 - val_split))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return imgs[train_idx], masks[train_idx], imgs[val_idx], masks[val_idx]


def split_by_patient(imgs, masks, val_split=0.2):
    patients = load_patient_nums()
    unique = list(set(patients.tolist()))
    random.shuffle(unique)
    cut = int(len(unique) * val_split)
    val_patients = set(unique[:cut])
    val_idx   = [i for i, p in enumerate(patients) if p in val_patients]
    train_idx = [i for i, p in enumerate(patients) if p not in val_patients]
    return imgs[train_idx], masks[train_idx], imgs[val_idx], masks[val_idx]


class SegDataset(Dataset):
    def __init__(self, imgs, masks, mean, std):
        self.imgs  = ((imgs.astype(np.float32) - mean) / std)
        self.masks = masks.astype(np.float32)
        self.nerve = nerve_presence(masks)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.imgs[i]),
            torch.from_numpy(self.masks[i]),
            torch.tensor(self.nerve[i]),
        )


def train_epoch(model, loader, optimizer, seg_w=1.0, aux_w=0.5):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    bce = nn.BCELoss()
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        seg, aux = model(imgs)
        loss = seg_w * dice_loss(seg, masks) + aux_w * bce(aux.reshape(-1), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
            total_dice += dice_coef(seg, masks).item()
    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    bce = nn.BCELoss()
    for imgs, masks, labels in loader:
        imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
        seg, aux = model(imgs)
        loss = dice_loss(seg, masks) + 0.5 * bce(aux.reshape(-1), labels.reshape(-1))
        total_loss += loss.item()
        total_dice += dice_coef(seg, masks).item()
    n = len(loader)
    return total_loss / n, total_dice / n


def train(model_name='resnet34_plus', split_random_flag=True, epochs=50, batch_size=64, lr=1e-4, val_split=0.2, patience=5):
    os.makedirs(RES_DIR, exist_ok=True)
    canonical = resolve_model_name(model_name)
    weights_path, history_path, val_eval_path, test_masks_path, test_probs_path = get_model_paths(canonical)

    spec = get_model_spec(canonical)
    print("\n" + "=" * 60)
    print(f" TRAINING MODEL: {spec['display_name']} [{spec['canonical_name']}]")
    print(f" Backbone:    {spec['backbone']}")
    print(f" Decoder:     {spec['decoder']}")
    print(f" Parameters:  {spec['total_params_fmt']}")
    print("=" * 60 + "\n")

    print(f'Loading dataset for model [{canonical}]...')
    imgs, masks = load_train()
    imgs  = resize_batch(imgs)
    masks = resize_batch(masks)

    split_fn = split_random if split_random_flag else split_by_patient
    x_tr, y_tr, x_val, y_val = split_fn(imgs, masks, val_split)

    mean, std = x_tr.astype(np.float32).mean(), x_tr.astype(np.float32).std()
    save_pickle(MEANSTD_PATH, (mean, std))

    print('Augmenting training data...')
    x_tr_f = x_tr.astype(np.float32)
    y_tr_f = (y_tr.astype(np.float32) / 255.0)
    x_tr_aug, y_tr_aug = augment(x_tr_f, y_tr_f)
    y_val_f = y_val.astype(np.float32) / 255.0

    train_ds = SegDataset(x_tr_aug, y_tr_aug, mean, std)
    val_ds   = SegDataset(x_val,    y_val_f,  mean, std)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f'Initializing model [{canonical}] on device [{DEVICE}]...')
    model     = get_model(canonical).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2)

    best_val_loss = float('inf')
    wait = 0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice = train_epoch(model, train_dl, optimizer)
        val_loss, val_dice = eval_epoch(model, val_dl)
        scheduler.step(val_loss)
        print(f'Epoch {epoch:03d}/{epochs:03d} | train loss: {tr_loss:.4f} dice: {tr_dice:.4f} | val loss: {val_loss:.4f} dice: {val_dice:.4f}')

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(val_dice)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stop triggered at epoch {epoch}')
                break

    save_pickle(history_path, history)
    print(f'\nTraining complete for [{canonical}]. Best val loss: {best_val_loss:.4f}')
    print(f'Saved model weights -> {weights_path}')

    # Evaluate best model on validation split and save ground truth + predictions for evaluation
    print(f'Generating validation evaluation payload for [{canonical}]...')
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    all_val_segs, all_val_auxs = [], []
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
    print(f'Saved validation evaluation metrics payload -> {val_eval_path}')

    return model, mean, std


def predict(model_name='resnet34_plus', mean=None, std=None, batch_size=128):
    from data import load_pickle
    canonical = resolve_model_name(model_name)
    weights_path, history_path, val_eval_path, test_masks_path, test_probs_path = get_model_paths(canonical)

    if mean is None:
        mean, std = load_pickle(MEANSTD_PATH)

    print(f'Loading test dataset for model [{canonical}]...')
    imgs_test = load_test()
    imgs_test = resize_batch(imgs_test)
    imgs_test = (imgs_test.astype(np.float32) - mean) / std

    model = get_model(canonical).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    all_masks, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(imgs_test), batch_size):
            batch = torch.from_numpy(imgs_test[i:i+batch_size]).to(DEVICE)
            seg, aux = model(batch)
            all_masks.append(seg.cpu().numpy())
            all_probs.append(aux.reshape(-1).cpu().numpy())

    masks = np.concatenate(all_masks, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    np.save(test_masks_path, masks)
    np.save(test_probs_path, probs)
    print(f'Test predictions saved to {test_masks_path} and {test_probs_path}')


if __name__ == '__main__':
    print("PyTorch CUDA available: " + str(torch.cuda.is_available()))
    valid_choices = sorted(list(set(list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys()))))
    
    parser = argparse.ArgumentParser(description="Train nerve segmentation models")
    parser.add_argument('--model', type=str, default='resnet34_plus', choices=valid_choices,
                        help='Model architecture to train (unet_inception, resnet34_plus, se_resnext50, etc.)')
    parser.add_argument('--split', type=str, default='random', choices=['random', 'patient'],
                        help='Validation split strategy')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--batch',  type=int, default=64, help='Batch size')
    parser.add_argument('--lr',     type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    args = parser.parse_args()

    model, mean, std = train(
        model_name=args.model,
        split_random_flag=(args.split == 'random'),
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
    )
    predict(model_name=args.model, mean=mean, std=std)


