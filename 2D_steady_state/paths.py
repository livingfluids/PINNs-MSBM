from pathlib import Path
import config

ROOT = Path(__file__).resolve().parent
data_dir = ROOT / "data" / config.DATA_DIR
visu_dir = ROOT / "visuals"