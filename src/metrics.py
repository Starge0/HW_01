"""Full-reference metrics, computed against the ORIGINAL image."""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(ref, img):
    return float(structural_similarity(ref, img, channel_axis=2, data_range=255))


def mae(ref, img):
    return float(np.mean(np.abs(ref.astype(np.float32) - img.astype(np.float32))))


def rms_contrast(img):
    """Std of the CIE-Lab L channel: a simple 'how contrasted does it look' indicator.
    Not a quality metric by itself; used to discuss the gap between numbers and appearance."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].std())


def all_metrics(ref, img):
    return {"PSNR": psnr(ref, img), "SSIM": ssim(ref, img),
            "MAE": mae(ref, img), "L_std": rms_contrast(img)}
