"""Synthetic degradation: contrast reduction (gain/bias) + Gaussian noise."""
import numpy as np
from . import config as C


def degrade(img_bgr, alpha=C.ALPHA, beta=C.BETA, sigma=C.SIGMA, seed=C.SEED):
    """g = alpha*f + beta + N(0, sigma^2), clipped to [0, 255].

    Work in float32: doing the arithmetic on uint8 would wrap around.
    A new generator is created from `seed` at every call, so the same noise is
    produced for the same image shape no matter in which order images are processed.
    """
    rng = np.random.default_rng(seed)
    out = img_bgr.astype(np.float32) * alpha + beta
    out += rng.normal(0.0, sigma, img_bgr.shape).astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
