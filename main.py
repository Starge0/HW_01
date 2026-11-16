"""HW1 - Image enhancement. Runs the whole pipeline:

    data/raw  ->  data/original (resized)  ->  data/degraded (contrast + noise)
              ->  results/processed/<image>/<method>.png
              ->  results/metrics.csv  +  results/figures/*.png

Usage:   python main.py
"""
import cv2
import pandas as pd

from src import config as C
from src.degrade import degrade
from src.methods import METHODS
from src.metrics import all_metrics
from src import viz


def prepare_originals():
    """Resize the full-resolution downloads to MAX_SIDE px (longest side)."""
    C.ORIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(C.RAW_DIR.glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        scale = C.MAX_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(C.ORIG_DIR / f"{p.stem}.png"), img)      # PNG: no extra compression loss
        print(f"[prepare] {p.name}: -> {img.shape[1]}x{img.shape[0]}")


def main():
    for d in (C.DEGR_DIR, C.PROC_DIR, C.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    prepare_originals()

    rows = []
    for path in sorted(C.ORIG_DIR.glob("*.png")):
        name = path.stem
        orig = cv2.imread(str(path), cv2.IMREAD_COLOR)

        degr = degrade(orig)
        cv2.imwrite(str(C.DEGR_DIR / f"{name}.png"), degr)

        results, metrics = {}, {}
        out_dir = C.PROC_DIR / name
        out_dir.mkdir(exist_ok=True)
        for m in C.METHOD_ORDER:
            results[m] = METHODS[m](degr)            # every method receives the SAME degraded image
            cv2.imwrite(str(out_dir / f"{m}.png"), results[m])
            metrics[m] = all_metrics(orig, results[m])
            rows.append({"image": name, "method": m, **metrics[m]})
        # reference row: the original itself (shows the L_std we are trying to recover)
        # (PSNR is undefined -> NaN, because the error against itself is exactly zero)
        ref = all_metrics(orig, orig)
        ref["PSNR"] = float("nan")
        rows.append({"image": name, "method": "original", **ref})

        viz.comparison_figure(name, orig, degr, results, metrics, C.FIG_DIR / f"{name}_comparison.png")
        viz.histogram_figure(name, orig, degr, results, C.FIG_DIR / f"{name}_histograms.png")
        viz.zoom_figure(name, orig, degr, results, C.FIG_DIR / f"{name}_zoom.png")
        print(f"[done] {name}")

    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS_CSV, index=False, float_format="%.4f")

    pd.set_option("display.width", 140)
    print("\n=== Metrics per image ===")
    print(df.round(3).to_string(index=False))
    print("\n=== Mean over images ===")
    mean = df[df.method != "original"].groupby("method")[["PSNR", "SSIM", "MAE", "L_std"]].mean()
    print(mean.reindex(C.METHOD_ORDER).round(3))
    print(f"\nMetrics saved to {C.METRICS_CSV}")


if __name__ == "__main__":
    main()
