"""Fixed experiment parameters. Every method and metric reads from here so the
comparison between methods always uses the same data and the same settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ORIG_DIR = ROOT / "data" / "original"
DEGR_DIR = ROOT / "data" / "degraded"
PROC_DIR = ROOT / "results" / "processed"
FIG_DIR = ROOT / "results" / "figures"
METRICS_CSV = ROOT / "results" / "metrics.csv"

# --- data preparation -------------------------------------------------------
MAX_SIDE = 800            # longest side (px) of the images used in the experiments

# --- synthetic degradation:  g = ALPHA * f + BETA + N(0, SIGMA^2) ------------
ALPHA = 0.5               # gain  (< 1 reduces contrast)
BETA = 40                 # bias  (raises the black level -> washed-out look)
SIGMA = 5.0               # std of the additive Gaussian noise (gray levels)
SEED = 42                 # fixed seed -> identical noise on every run

# --- enhancement methods ----------------------------------------------------
STRETCH_P_LOW, STRETCH_P_HIGH = 1, 99   # percentiles used by linear stretching
GAMMA = 1.5                             # convention of the course: g = f^(1/gamma)
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
NLM_H = 5                 # non-local-means strength (luminance)
NLM_H_COLOR = 5           # non-local-means strength (color)
NLM_TEMPLATE = 7
NLM_SEARCH = 21

# Methods in the order they appear in tables and figures.
METHOD_ORDER = ["baseline", "stretch_channel", "stretch_luminance",
                "he_lab", "he_hsv", "gamma_lab", "clahe_nlm"]
