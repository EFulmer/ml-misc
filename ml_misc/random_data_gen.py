"""Helpers for generating random data."""
# TODO: add helpers for generating e.g. randomized DataFrames with
# randomized column names, etc.
from typing import Optional

import numpy as np


def rand_data(
    shape: tuple[int, int], *, rng: np.random._generator.Generator=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate randomized X and y with the shape given.
    X is taken from [0.0, 1.0), and y is taken from Uniform(0, 50).

    An rng can optionally be passed in, otherwise the standard NumPy RNG
    seeded with 0 is used.

    Args:
        shape: Tuple determining the shape of the random X and y to be
            generated.
        rng: Optional RNG for generating X and y.

    Returns:
        X and y, as described above.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = shape[0]
    X = rng.random(size=shape)
    y = rng.uniform(0, 50, size=(n, 1))
    return X, y
