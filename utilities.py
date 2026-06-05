import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
import os 
from skimage import io
from PIL import Image
import albumentations as A
from scipy.ndimage import gaussian_filter
from tensorflow.keras import backend as K

# ─── Augmentation Configuration ───────────────────────────────────────────────
# Set individual flags to True/False to enable or disable each augmentation.
# This config is passed to augment_mri() and can be overridden at training time.
AUGMENTATION_CONFIG = {
    "enabled": True,
    "horizontal_flip": True,       # Flip left-right (p=0.5)
    "vertical_flip": True,         # Flip top-bottom (p=0.3)
    "rotation_limit": 15,          # Random rotate ±15° (p=0.5)
    "brightness_contrast": True,   # Random brightness/contrast shift (p=0.5)
    "gaussian_noise": True,        # Simulate scanner noise (p=0.4)
    "elastic_transform": True,     # Simulate tissue deformation (p=0.3)
    "bias_field": True,            # Simulate MRI bias field artifact (p=0.4)
    "gamma_correction": True,      # Random gamma shift for contrast variety (p=0.4)
}


def augment_mri(img, mask, config=None):
    """
    Apply MRI-specific augmentations to an image and its corresponding mask.

    Augmentations are applied consistently to both image and mask to preserve
    spatial alignment. The bias field augmentation is applied to the image only,
    as it simulates scanner intensity non-uniformity and does not affect anatomy.

    Parameters
    ----------
    img  : np.ndarray  Shape (H, W, 3), float64, raw pixel values 0–255.
    mask : np.ndarray  Shape (H, W),    float64, raw pixel values 0–255.
    config : dict or None
        Augmentation flags. Defaults to module-level AUGMENTATION_CONFIG.

    Returns
    -------
    img_aug  : np.ndarray  Shape (H, W, 3), float64
    mask_aug : np.ndarray  Shape (H, W),    float64
    """
    if config is None:
        config = AUGMENTATION_CONFIG

    if not config.get("enabled", True):
        return img, mask

    aug_list = []

    if config.get("horizontal_flip"):
        aug_list.append(A.HorizontalFlip(p=0.5))

    if config.get("vertical_flip"):
        aug_list.append(A.VerticalFlip(p=0.3))

    if config.get("rotation_limit"):
        aug_list.append(A.Rotate(limit=config["rotation_limit"], p=0.5))

    if config.get("brightness_contrast"):
        aug_list.append(A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5))

    if config.get("elastic_transform"):
        aug_list.append(A.ElasticTransform(
            alpha=1, sigma=50, p=0.3))

    if config.get("gaussian_noise"):
        aug_list.append(A.GaussNoise(std_range=(0.04, 0.22), p=0.4))

    if config.get("gamma_correction"):
        aug_list.append(A.RandomGamma(gamma_limit=(80, 120), p=0.4))

    pipeline = A.Compose(aug_list)

    # Albumentations expects uint8; convert and back
    img_uint8  = np.clip(img,  0, 255).astype(np.uint8)
    mask_uint8 = np.clip(mask, 0, 255).astype(np.uint8)

    augmented = pipeline(image=img_uint8, mask=mask_uint8)
    img_aug  = augmented["image"].astype(np.float64)
    mask_aug = augmented["mask"].astype(np.float64)

    # Bias field simulation (MRI-specific, image only)
    # Models a smooth low-frequency intensity variation caused by RF coil
    # non-uniformity — a common artefact across different MRI scanners.
    if config.get("bias_field") and np.random.rand() < 0.4:
        bias = gaussian_filter(
            np.random.randn(*img_aug.shape[:2]), sigma=30)
        bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)
        bias = 1.0 + 0.3 * bias                          # scale to [1.0, 1.3]
        img_aug = img_aug * bias[:, :, np.newaxis]

    return img_aug, mask_aug


# ─── Data Generator ───────────────────────────────────────────────────────────
class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras data generator for MRI segmentation.

    Parameters
    ----------
    augment : bool
        If True, apply augment_mri() to every training sample.
        Should be False for validation/test generators.
    aug_config : dict or None
        Override for AUGMENTATION_CONFIG. Pass None to use defaults.
    """

    def __init__(self, ids, mask, image_dir='./', batch_size=16,
                 img_h=256, img_w=256, shuffle=True,
                 augment=False, aug_config=None):
        self.ids        = ids
        self.mask       = mask
        self.image_dir  = image_dir
        self.batch_size = batch_size
        self.img_h      = img_h
        self.img_w      = img_w
        self.shuffle    = shuffle
        self.augment    = augment
        self.aug_config = aug_config if aug_config is not None else AUGMENTATION_CONFIG
        self.on_epoch_end()

    def __len__(self):
        'Get the number of batches per epoch'
        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):
        'Generate a batch of data'
        indexes   = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        list_ids  = [self.ids[i]  for i in indexes]
        list_mask = [self.mask[i] for i in indexes]
        X, y = self.__data_generation(list_ids, list_mask)
        return X, y

    def on_epoch_end(self):
        'Update indices after each epoch'
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):
        'Generate one batch of data'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        for i in range(len(list_ids)):
            img_path  = './' + str(list_ids[i])
            mask_path = './' + str(list_mask[i])

            img  = io.imread(img_path)
            mask = io.imread(mask_path)

            # Resize
            img  = cv2.resize(img,  (self.img_h, self.img_w))
            img  = np.array(img,  dtype=np.float64)
            mask = cv2.resize(mask, (self.img_h, self.img_w))
            mask = np.array(mask, dtype=np.float64)

            # Apply MRI augmentation before standardisation (training only)
            if self.augment:
                img, mask = augment_mri(img, mask, config=self.aug_config)

            # Standardise
            img -= img.mean()
            img /= img.std()

            mask -= mask.mean()
            mask /= (mask.std() + 1e-8)   # guard against zero-std blank masks

            X[i,] = img
            y[i,] = np.expand_dims(mask, axis=2)

        # Binarise mask
        y = (y > 0).astype(int)
        return X, y


# ─── Prediction ───────────────────────────────────────────────────────────────
def prediction(test, model, model_seg):
    """
    Two-stage prediction: classification → segmentation.

    If the classifier is ≥99 % confident there is no tumour the image is
    labelled as no-defect without running the segmentation model.
    """
    directory = "./"
    mask = []
    image_id = []
    has_mask = []

    for i in test.image_path:
        path = directory + str(i)
        img  = io.imread(path)
        img  = img * 1. / 255.
        img  = cv2.resize(img, (256, 256))
        img  = np.array(img, dtype=np.float64)
        img  = np.reshape(img, (1, 256, 256, 3))

        is_defect = model.predict(img)

        if np.argmax(is_defect) == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask')
            continue

        img = io.imread(path)
        X   = np.empty((1, 256, 256, 3))
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)
        img -= img.mean()
        img /= img.std()
        X[0,] = img

        predict = model_seg.predict(X)

        if predict.round().astype(int).sum() == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append('No mask')
        else:
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)

    return image_id, mask, has_mask


# ─── Loss Functions ───────────────────────────────────────────────────────────
'''
Custom loss functions from:
https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

@article{focal-unet,
  title={A novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}
'''
def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos   = K.sum(y_true_pos * y_pred_pos)
    false_neg  = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos  = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true, y_pred):
    pt_1  = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f     = K.flatten(y_true)
    y_pred_f     = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union        = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

def sensitivity(y_true, y_pred):
    true_positives    = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives    = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def precision_metric(y_true, y_pred):
    true_positives    = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


# ─── Test-Time Augmentation (TTA) ─────────────────────────────────────────────
def predict_with_tta_classification(model, img):
    pred       = model.predict(img)
    img_flip   = np.flip(img,  axis=2)
    pred_flip  = model.predict(img_flip)
    img_vflip  = np.flip(img,  axis=1)
    pred_vflip = model.predict(img_vflip)
    return (pred + pred_flip + pred_vflip) / 3

def predict_with_tta_segmentation(model, img):
    pred       = model.predict(img)
    img_flip   = np.flip(img,  axis=2)
    pred_flip  = model.predict(img_flip)
    pred_flip  = np.flip(pred_flip,  axis=2)
    img_vflip  = np.flip(img,  axis=1)
    pred_vflip = model.predict(img_vflip)
    pred_vflip = np.flip(pred_vflip, axis=1)
    return (pred + pred_flip + pred_vflip) / 3
