import numpy as np

FAST_MODE = True
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_PREFIX = "ldpc"

RANDOM_SEED = 12
INFORMATION_BITS = 1024 if FAST_MODE else 2048

ITERATIONS = [1, 2, 3, 4, 5, 6, 7]
LDPC_EBN0_DB = np.array([-1.0, 0.0, 0.8, 1.0, 1.15, 1.25, 1.30], dtype=float) if FAST_MODE else np.arange(-1.0, 1.31, 0.15)

MIN_FRAMES = 30 if FAST_MODE else 180
MAX_FRAMES = 180 if FAST_MODE else 1200
TARGET_ERRORS = 120 if FAST_MODE else 500
BENCHMARK_BLOCKS = 8 if FAST_MODE else 20

SUPPORTED_CODE_RATES = {"1/3": 1/3, "1/2": 1/2, "3/4": 3/4, "7/8": 7/8}
SELECTED_CODE_RATE_LABEL = "1/3"

def sigma2_from_ebn0(ebn0_db, code_rate):
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0)
