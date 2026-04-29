"""Project paths and global configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "dataset" / "Datasetv2.csv"
RESULT_DIR = ROOT_DIR / "result"
GRAPH_DIR = RESULT_DIR / "graph"
REPORT_DIR = RESULT_DIR / "report"

RANDOM_STATE = 42

PLOT_DPI = 130
FIG_SIZE_DEFAULT = (9, 5)
FIG_SIZE_WIDE = (11, 5)
FIG_SIZE_TALL = (9, 6)
