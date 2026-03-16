import numpy as np

FAST_MODE = True
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_PREFIX = "turbo"

RANDOM_SEED = 12

INFORMATION_BITS = 1024 if FAST_MODE else 2048
ITERATIONS = [1, 2, 3, 4, 5, 6, 7]

# Convolutional baseline sweep
CONV_EBN0_DB = np.array([-8.5, -6.8, -4.8, -2.8, -0.8, 1.5, 3.5], dtype=float)

# Turbo waterfall sweep
TURBO_EBN0_DB = np.array([-1.0, 0.0, 0.8, 1.0, 1.15, 1.25, 1.30], dtype=float) if FAST_MODE else np.arange(-1.0, 1.31, 0.15)

CONV_MIN_FRAMES = 40 if FAST_MODE else 160
CONV_MAX_FRAMES = 200 if FAST_MODE else 1200
CONV_TARGET_ERRORS = 120 if FAST_MODE else 500

MIN_FRAMES = 30 if FAST_MODE else 160
MAX_FRAMES = 150 if FAST_MODE else 900
TARGET_ERRORS = 120 if FAST_MODE else 500

BENCHMARK_BLOCKS = 8 if FAST_MODE else 20

RSC_MEMORY = 2
RSC_TAIL_LENGTH = 2
RSC_STATE_COUNT = 2 ** RSC_MEMORY

TRACEBACK_DEPTH = 15

SUPPORTED_CODE_RATES = {
    "1/3": 1.0 / 3.0,
    "1/2": 1.0 / 2.0,
    "3/4": 3.0 / 4.0,
    "7/8": 7.0 / 8.0,
}
SELECTED_CODE_RATE_LABEL = "1/3"

def sigma2_from_ebn0(ebn0_db: float, code_rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * code_rate * ebn0)

def get_puncture_definition(rate_label: str):
    if rate_label == "1/3":
        return np.array([1], dtype=np.int8), np.array([1], dtype=np.int8)
    if rate_label == "1/2":
        return np.array([1, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8)
    if rate_label == "3/4":
        return np.array([1, 0, 0, 1], dtype=np.int8), np.array([0, 1, 1, 0], dtype=np.int8)
    if rate_label == "7/8":
        return np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=np.int8), np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    raise ValueError(rate_label)
