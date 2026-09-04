import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from data import load_test, load_pickle
from model import (
    get_model,
    get_model_spec,
    resolve_model_name,
    MODEL_REGISTRY,
    MODEL_ALIASES,
    IMG_ROWS,
    IMG_COLS,
)
from train import resize_batch, get_model_paths, MEANSTD_PATH

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORMS = [
    ('flip_h',   lambda x: x.flip(-1),         lambda x: x.flip(-1)),
    ('flip_v',   lambda x: x.flip(-2),         lambda x: x.flip(-2)),
    ('zoom_in',  lambda x: _zoom(x, 1.05),     lambda x: _zoom(x, 1 / 1.05)),
    ('zoom_out', lambda x: _zoom(x, 0.95),     lambda x: _zoom(x, 1 / 0.95)),
    ('shift',    lambda x: x + 5.0 / (x.std() + 1e-6), lambda x: x),
]


def _zoom(x, factor):
    h, w = x.shape[-2], x.shape[-1]
    new_h, new_w = int(h * factor), int(w * factor)
    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


@torch.no_grad()
def _predict_batch(model, batch, batch_size=128):
    segs, auxs = [], []
    for i in range(0, len(batch), batch_size):
        chunk = batch[i:i + batch_size].to(DEVICE)
        s, a = model(chunk)
        segs.append(s.cpu())
        auxs.append(a.reshape(-1).cpu())
    return torch.cat(segs), torch.cat(auxs)


def predict_model_tta(model_name, imgs_tensor, batch_size=128, use_tta=True):
    canonical = resolve_model_name(model_name)
    weights_path, history_path, val_eval_path, test_masks_path, test_probs_path = get_model_paths(canonical)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: '{weights_path}'.\n"
            f"Please train the model first by running: python train.py --model {canonical}"
        )

    spec = get_model_spec(canonical)
    print("\n" + "-" * 60)
    print(f" INFERENCE MODEL SPECIFICATION: {spec['display_name']} [{spec['canonical_name']}]")
    print(f" Architecture: {spec['description']}")
    print(f" Backbone:     {spec['backbone']}")
    print(f" Decoder:      {spec['decoder']}")
    print(f" Total Params: {spec['total_params_fmt']} ({spec['total_params']:,})")
    print(f" Weights File: {weights_path}")
    print("-" * 60 + "\n")

    model = get_model(canonical).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    print(f"Running model forward pass (TTA={'ON' if use_tta else 'OFF'})...")
    pred_masks, pred_probs = _predict_batch(model, imgs_tensor, batch_size)

    if use_tta:
        aug_masks = [pred_masks]
        aug_probs = [pred_probs]
        for name, do_t, undo_t in TRANSFORMS:
            t_imgs = do_t(imgs_tensor)
            s, a = _predict_batch(model, t_imgs, batch_size)
            aug_masks.append(undo_t(s))
            aug_probs.append(a)
        final_mask = torch.stack(aug_masks).mean(0).numpy()
        final_prob = torch.stack(aug_probs).mean(0).numpy()
    else:
        final_mask = pred_masks.numpy()
        final_prob = pred_probs.numpy()

    np.save(test_masks_path, final_mask)
    np.save(test_probs_path, final_prob)
    
    nerve_count = int((final_prob >= 0.5).sum())
    total_samples = len(final_prob)
    print(f"Inference summary for [{canonical}]:")
    print(f"  Test samples:     {total_samples}")
    print(f"  Nerve detected:   {nerve_count} ({nerve_count / total_samples * 100:.1f}%)")
    print(f"  Mean nerve prob:  {final_prob.mean():.4f}")
    print(f"  Saved outputs -> {test_masks_path} & {test_probs_path}")

    return final_mask, final_prob


def run_prediction(model_name='resnet34_plus', batch_size=128, models_to_ensemble=None, use_tta=True):
    if not os.path.exists(MEANSTD_PATH):
        raise FileNotFoundError(f"Normalization file '{MEANSTD_PATH}' missing. Please run 'python train.py' first.")

    mean, std = load_pickle(MEANSTD_PATH)

    print('Loading test dataset...')
    imgs_test = load_test()
    imgs_test = resize_batch(imgs_test)
    imgs_test = (imgs_test.astype('float32') - mean) / std
    imgs_tensor = torch.from_numpy(imgs_test)

    if model_name.lower() == 'ensemble':
        default_ensemble = ['resnet34_plus', 'unet_inception', 'se_resnext50']
        target_models = [resolve_model_name(m) for m in (models_to_ensemble or default_ensemble)]
        
        print("\n" + "=" * 60)
        print(f" ENSEMBLE INFERENCE ACROSS {len(target_models)} MODELS: {target_models}")
        print("=" * 60)
        
        all_masks, all_probs = [], []
        for m in target_models:
            m_mask, m_prob = predict_model_tta(m, imgs_tensor, batch_size, use_tta=use_tta)
            all_masks.append(m_mask)
            all_probs.append(m_prob)

        ens_mask = np.mean(all_masks, axis=0)
        ens_prob = np.mean(all_probs, axis=0)

        _, _, _, ens_mask_path, ens_prob_path = get_model_paths('ensemble')
        np.save(ens_mask_path, ens_mask)
        np.save(ens_prob_path, ens_prob)

        nerve_count = int((ens_prob >= 0.5).sum())
        total_samples = len(ens_prob)
        print("\n" + "=" * 60)
        print(f" ENSEMBLE SUMMARY:")
        print(f"  Test samples:     {total_samples}")
        print(f"  Nerve detected:   {nerve_count} ({nerve_count / total_samples * 100:.1f}%)")
        print(f"  Ensemble saved -> {ens_mask_path} & {ens_prob_path}")
        print("=" * 60 + "\n")
    else:
        canonical = resolve_model_name(model_name)
        predict_model_tta(canonical, imgs_tensor, batch_size, use_tta=use_tta)


if __name__ == '__main__':
    valid_models = sorted(list(set(list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys()))))
    parser = argparse.ArgumentParser(description="Predict nerve segmentation on test dataset")
    parser.add_argument('--model', type=str, default='resnet34_plus',
                        choices=valid_models + ['ensemble'],
                        help='Model architecture to predict with, or "ensemble"')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for prediction')
    parser.add_argument('--ensemble_models', nargs='*', default=['resnet34_plus', 'unet_inception', 'se_resnext50'],
                        help='Models to combine when --model ensemble is specified')
    parser.add_argument('--no_tta', action='store_true', help='Disable test-time augmentation')
    args = parser.parse_args()

    run_prediction(
        model_name=args.model,
        batch_size=args.batch,
        models_to_ensemble=args.ensemble_models,
        use_tta=not args.no_tta
    )


