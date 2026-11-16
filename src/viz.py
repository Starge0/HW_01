"""Figures: side-by-side comparison, luminance histograms, zoomed crops (noise / detail)."""
import cv2
import matplotlib
matplotlib.use("Agg")                      # no window: the pipeline runs headless
import matplotlib.pyplot as plt
import numpy as np
from . import config as C

LABELS = {
    "baseline": "Degraded (baseline)",
    "stretch_channel": "Stretch per channel",
    "stretch_luminance": "Stretch on L (Lab)",
    "he_lab": "Hist. eq. on L (Lab)",
    "he_hsv": "Hist. eq. on V (HSV)",
    "gamma_lab": f"Gamma on L (g={C.GAMMA})",
    "clahe_nlm": "CLAHE + NLM (extra)",
}


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # matplotlib expects RGB


def comparison_figure(name, orig, degr, results, metrics, path):
    """Original | degraded | six methods. Same size and 0-255 scale for every panel."""
    methods = [m for m in C.METHOD_ORDER if m != "baseline"]
    panels = [("Original (reference)", orig), (LABELS["baseline"], degr)]
    for m in methods:
        mt = metrics[m]
        panels.append((f"{LABELS[m]}\nPSNR {mt['PSNR']:.1f} dB | SSIM {mt['SSIM']:.3f}", results[m]))
    mt = metrics["baseline"]
    panels[1] = (f"{LABELS['baseline']}\nPSNR {mt['PSNR']:.1f} dB | SSIM {mt['SSIM']:.3f}", degr)

    h, w = orig.shape[:2]
    cols = 4 if w >= h else 4
    rows = int(np.ceil(len(panels) / cols))
    pw = 4.0                                   # panel width (in): identical for every panel
    ph = pw * h / w + 0.85                     # + room for the two-line title
    fig, axes = plt.subplots(rows, cols, figsize=(pw * cols, ph * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (title, im) in zip(np.ravel(axes), panels):
        ax.imshow(_rgb(im), vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def histogram_figure(name, orig, degr, results, path):
    """Histogram of the Lab luminance for the original, the degraded and every method."""
    def L(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].ravel()
    entries = [("Original", orig)] + [(LABELS[m], results[m]) for m in C.METHOD_ORDER]
    cols = 4
    rows = int(np.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), sharex=True)
    for ax in np.ravel(axes):
        ax.axis("off")
    ymax = 0
    hists = []
    for _, im in entries:
        hh, _ = np.histogram(L(im), bins=64, range=(0, 255))
        hists.append(hh)
        ymax = max(ymax, hh.max())
    for ax, (title, _), hh in zip(np.ravel(axes), entries, hists):
        ax.axis("on")
        ax.bar(np.linspace(0, 255, 64, endpoint=False), hh, width=255 / 64, color="0.3")
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    fig.suptitle(f"{name} - histogram of L (CIE-Lab)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pick_patches(orig, size=96, stride=48):
    """Deterministic choice of two crops from the ORIGINAL: flattest block (shows noise
    amplification) and most textured block (shows loss of detail)."""
    L = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    h, w = L.shape
    best_flat, best_det = (1e9, (0, 0)), (-1, (0, 0))
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            blk = L[y:y + size, x:x + size]
            s = blk.std()
            m = blk.mean()
            if 40 < m < 215 and s < best_flat[0]:       # flat but not saturated black/white
                best_flat = (s, (y, x))
            if s > best_det[0]:
                best_det = (s, (y, x))
    return best_flat[1], best_det[1], size


def zoom_figure(name, orig, degr, results, path):
    flat, det, size = pick_patches(orig)
    entries = [("Original", orig)] + [(LABELS[m].split(" (")[0], results[m]) for m in C.METHOD_ORDER]
    fig, axes = plt.subplots(2, len(entries), figsize=(2.1 * len(entries), 4.6))
    for r, ((y, x), tag) in enumerate([(flat, "flat area"), (det, "textured area")]):
        for c, (title, im) in enumerate(entries):
            ax = axes[r, c]
            ax.imshow(_rgb(im[y:y + size, x:x + size]), vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=7)
            if c == 0:
                ax.set_ylabel(tag, fontsize=8)
    fig.suptitle(f"{name} - zoom {size}x{size} px", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
