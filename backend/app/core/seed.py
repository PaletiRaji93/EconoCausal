"""Reproducibility helper: seeds Python and NumPy global RNGs."""
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed global RNGs and return a fresh Generator for explicit use."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
