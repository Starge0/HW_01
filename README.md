# HW1 - Image enhancement (Computer Vision, Fall 2026)

Classical (no deep learning) enhancement of images with low contrast, non-uniform illumination
and noise. Five enhancement strategies are compared against a baseline, qualitatively and
with PSNR / SSIM / MAE.

## Requirements
- Python 3.10+ (developed and tested with Python 3.12)
- `pip install -r requirements.txt`  ->  numpy, opencv-python, scikit-image, matplotlib, pandas

## Run
```bash
python main.py
```
One command runs the whole pipeline from `data/raw/` and writes everything under `data/` and
`results/`. No window is opened. It takes well under a minute on a laptop CPU.

## What the pipeline does
1. **Prepare**: resizes `data/raw/*.jpg` to 800 px (longest side) -> `data/original/*.png`.
2. **Degrade**: `g = 0.5 * f + 40 + N(0, 5^2)`, clipped to [0, 255] -> `data/degraded/*.png`.
3. **Enhance** the *same* degraded image with every method -> `results/processed/<image>/<method>.png`.
4. **Evaluate** each result against the **original** -> `results/metrics.csv`.
5. **Figures** -> `results/figures/` (comparison, histograms, zoomed crops).

The originals are the reference **only because the degradation is synthetic**. With real
degraded photos a clean reference usually does not exist, so full-reference metrics such as
PSNR/SSIM cannot be computed.

## Methods (`src/methods.py`)
| Key | Method |
|---|---|
| `baseline` | degraded image, no processing |
| `stretch_channel` | linear stretch of the 1st-99th percentile, **per B/G/R channel** |
| `stretch_luminance` | same stretch, only on **L of CIE-Lab** (a, b untouched) |
| `he_lab` | histogram equalization on **L of CIE-Lab** |
| `he_hsv` | histogram equalization on **V of HSV** |
| `gamma_lab` | gamma on L, `g = f^(1/gamma)`, gamma = 1.5 (convention of the course) |
| `clahe_nlm` | **extra method**: CLAHE on L (clip 2.0, 8x8 tiles) + non-local-means denoising (h = 5) |

`he_lab(img, alpha)` also supports the *partial* equalization `alpha*c(I) + (1-alpha)*I`.

## Fixed parameters (`src/config.py`)
Degradation: `ALPHA=0.5`, `BETA=40`, `SIGMA=5`, **`SEED=42`** (a new `numpy.random.default_rng(42)`
is created for each image, so the noise is identical on every run).
Everything else (percentiles, gamma, CLAHE, NLM) is also defined there and is the same for all images.

## File structure
```
main.py                  entry point
requirements.txt
SOURCES.md               source, author and license of every image
src/
  config.py              all parameters and paths
  degrade.py             synthetic degradation
  methods.py             enhancement methods
  metrics.py             PSNR, SSIM, MAE, L_std
  viz.py                 figures
data/raw/                downloaded photographs (full resolution)
data/original/           resized references
data/degraded/           degraded inputs
results/processed/       one folder per image, one PNG per method
results/figures/         *_comparison.png, *_histograms.png, *_zoom.png
results/metrics.csv      image, method, PSNR, SSIM, MAE, L_std
```

## Metrics
`PSNR` (dB, higher is better), `SSIM` (0-1, higher is better), `MAE` (gray levels, lower is better),
all against the original. `L_std` is the standard deviation of the Lab luminance: it is **not** a
quality metric, only an indicator of how contrasted the image is (it is also reported for the
original, row `method = original`, whose PSNR is undefined and stored as NaN).
