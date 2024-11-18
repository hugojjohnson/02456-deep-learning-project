import sys
def set_random_seeds():
    # Python random module
    if 'random' in sys.modules:
        import random
        random.seed(42)
        print("Python random seed set.")
    # else:
    #     print("Python random module not available.")
    # NumPy
    if 'numpy' in sys.modules:
        import numpy as np
        np.random.seed(42)
        print("NumPy random seed set.")
    # else:
    #     print("NumPy not available.")
    # PyTorch (both CPU and GPU)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        print("PyTorch random seed set.")
    # else:
    #     print("PyTorch not available.")

# =========================================
# This was included in CCDS.

# from pathlib import Path

# from dotenv import load_dotenv
# from loguru import logger

# # Load environment variables from .env file if it exists
# load_dotenv()

# # Paths
# PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# DATA_DIR = PROJ_ROOT / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# INTERIM_DATA_DIR = DATA_DIR / "interim"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"
# EXTERNAL_DATA_DIR = DATA_DIR / "external"

# MODELS_DIR = PROJ_ROOT / "models"

# REPORTS_DIR = PROJ_ROOT / "reports"
# FIGURES_DIR = REPORTS_DIR / "figures"

# # If tqdm is installed, configure loguru with tqdm.write
# # https://github.com/Delgan/loguru/issues/135
# try:
#     from tqdm import tqdm

#     logger.remove(0)
#     logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
# except ModuleNotFoundError:
#     pass
