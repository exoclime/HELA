
from collections import namedtuple
import logging
from typing import Union, Iterable

import numpy as np

from .wpercentile import wpercentile

__all__ = [
    "Posterior",
    "resample_posterior",
    "posterior_percentile"
]

LOGGER = logging.getLogger(__name__)

# Posteriors are represented as a collection of weighted samples
Posterior = namedtuple("Posterior", ["samples", "weights"])


def resample_posterior(posterior: Posterior, num_draws: int) -> Posterior:

    p = posterior.weights / posterior.weights.sum()
    indices = np.random.choice(len(posterior.samples), size=num_draws, p=p)

    new_weights = np.bincount(indices, minlength=len(posterior.samples))
    mask = new_weights != 0
    new_samples = posterior.samples[mask]
    new_weights = posterior.weights[mask]

    return Posterior(new_samples, new_weights)


def posterior_percentile(
        posterior: Posterior,
        percentiles: Union[float, Iterable[float]]) -> np.ndarray:

    samples, weights = posterior
    return wpercentile(samples, weights, percentiles, axis=0)
