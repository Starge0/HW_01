"""Enhancement methods. All take and return an 8-bit BGR image (OpenCV convention).

Color images are never processed channel by channel, except in `stretch_channel`,
which is kept on purpose to show the color shift it produces.
"""
import cv2
import numpy as np
from . import config as C


# ---------- helpers ---------------------------------------------------------
def _linear_stretch(x, p_low, p_high):
    """Map the [p_low, p_high] percentiles of x onto [0, 255] (values outside are clipped)."""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (p_low, p_high))
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)


def _apply_on_L(img_bgr, fn):
    """Apply fn (uint8 2-D -> uint8 2-D) to the L channel of CIE-Lab, keep a and b."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = fn(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------- 1. baseline -----------------------------------------------------
def baseline(img):
    return img.copy()


# ---------- 2. linear contrast stretching -----------------------------------
def stretch_channel(img, p_low=C.STRETCH_P_LOW, p_high=C.STRETCH_P_HIGH):
    """Stretch B, G, R independently. Changes the ratios between channels -> color shift."""
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = _linear_stretch(img[..., c], p_low, p_high).astype(np.uint8)
    return out


def stretch_luminance(img, p_low=C.STRETCH_P_LOW, p_high=C.STRETCH_P_HIGH):
    """Stretch only L of CIE-Lab; a and b (the chromaticity) are untouched."""
    return _apply_on_L(img, lambda L: _linear_stretch(L, p_low, p_high).astype(np.uint8))


# ---------- 3. histogram equalization ---------------------------------------
def he_lab(img, alpha=1.0):
    """Equalize L of CIE-Lab. alpha<1 gives the 'partial' equalization
    f(I) = alpha*c(I) + (1-alpha)*I of the course slides."""
    def fn(L):
        eq = cv2.equalizeHist(L).astype(np.float32)
        return (alpha * eq + (1 - alpha) * L.astype(np.float32)).astype(np.uint8)
    return _apply_on_L(img, fn)


def he_hsv(img):
    """Equalize V of HSV (hue and saturation untouched)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------- 4. additional methods -------------------------------------------
def gamma_lab(img, gamma=C.GAMMA):
    """Gamma correction on L, convention of the slides: g = f^(1/gamma) (gamma>1 brightens)."""
    def fn(L):
        return (np.power(L.astype(np.float32) / 255.0, 1.0 / gamma) * 255.0).astype(np.uint8)
    return _apply_on_L(img, fn)


def clahe_lab(img, clip=C.CLAHE_CLIP, tile=C.CLAHE_TILE):
    """Contrast-limited adaptive histogram equalization on L (blocks of the slides)."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _apply_on_L(img, clahe.apply)


def clahe_nlm(img):
    """EXTRA METHOD: CLAHE followed by non-local-means denoising.

    CLAHE restores local contrast (good against uneven illumination) but amplifies
    noise; NLM removes that noise at the price of smoothing fine texture.
    """
    out = clahe_lab(img)
    return cv2.fastNlMeansDenoisingColored(out, None, C.NLM_H, C.NLM_H_COLOR,
                                           C.NLM_TEMPLATE, C.NLM_SEARCH)


METHODS = {
    "baseline": baseline,
    "stretch_channel": stretch_channel,
    "stretch_luminance": stretch_luminance,
    "he_lab": he_lab,
    "he_hsv": he_hsv,
    "gamma_lab": gamma_lab,
    "clahe_nlm": clahe_nlm,
}
