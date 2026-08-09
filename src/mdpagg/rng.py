from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Streams:

    problem: np.random.Generator
    sampling: np.random.Generator


def streams(master: int) -> Streams:
    problem, sampling = np.random.SeedSequence(master).spawn(2)
    return Streams(np.random.default_rng(problem), np.random.default_rng(sampling))
