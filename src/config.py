from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories (MATCH YOUR NOTEBOOK FOLDERS)
DATA_DIR = BASE_DIR / "Data"

RAW_DATA_DIR = DATA_DIR / "Raw_data"
PROCESSED_DATA_DIR = DATA_DIR / "Processed_data"

ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Model / preprocessing parameters
MAX_WORDS = 10000
MAX_LEN = 100

BATCH_SIZE = 64
EPOCHS = 10
