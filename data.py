import os
import numpy as np
import cv2
import pickle
from imge_process import preprocess_image

DATA_DIR = os.path.join(os.path.dirname(__file__), './')
NP_DIR  = os.path.join(os.path.dirname(__file__), 'np_data')
RAW_ROWS, RAW_COLS = 420, 580


def _path(name):
    return os.path.join(NP_DIR, name)


def prepare_data():
    os.makedirs(NP_DIR, exist_ok=True)
    _build_train()
    _build_test()
    print('Done.')


def _build_train():
    folder = os.path.join(DATA_DIR, 'train')
    names  = [f for f in os.listdir(folder) if 'mask' not in f]
    n      = len(names)
    imgs     = np.zeros((n, 1, RAW_ROWS, RAW_COLS), dtype=np.uint8)
    masks    = np.zeros((n, 1, RAW_ROWS, RAW_COLS), dtype=np.uint8)
    patients = np.zeros(n, dtype=np.uint8)
    for i, name in enumerate(names):
        mask_name   = name.split('.')[0] + '_mask.tif'
        img_raw     = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
        mask_raw    = cv2.imread(os.path.join(folder, mask_name), cv2.IMREAD_GRAYSCALE)

        imgs[i, 0]  = preprocess_image(img_raw)
        masks[i, 0] = mask_raw
        patients[i] = int(name.split('_')[0])
        if i % 100 == 0:
            print(f'  train {i}/{n}')
    np.save(_path('imgs_train.npy'),       imgs)
    np.save(_path('imgs_mask_train.npy'),  masks)
    np.save(_path('imgs_patient.npy'),     patients)


def _build_test():
    folder = os.path.join(DATA_DIR, 'test')
    names  = os.listdir(folder)
    n      = len(names)
    imgs = np.zeros((n, 1, RAW_ROWS, RAW_COLS), dtype=np.uint8)
    ids  = np.zeros(n, dtype=np.int32)
    for i, name in enumerate(names):
        img_raw    = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
        imgs[i, 0] = preprocess_image(img_raw)
        ids[i]     = int(name.split('.')[0])
        if i % 100 == 0:
            print(f'  test {i}/{n}')
    np.save(_path('imgs_test.npy'),    imgs)
    np.save(_path('imgs_id_test.npy'), ids)


def _check_exists(filename):
    p = _path(filename)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required data file '{p}' not found. Please run 'python data.py' to generate binary datasets from raw images."
        )
    return p


def load_train():
    return np.load(_check_exists('imgs_train.npy')), np.load(_check_exists('imgs_mask_train.npy'))


def load_test():
    return np.load(_check_exists('imgs_test.npy'))


def load_test_ids():
    return np.load(_check_exists('imgs_id_test.npy'))


def load_patient_nums():
    return np.load(_check_exists('imgs_patient.npy'))


def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: '{path}'.")
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    prepare_data()

