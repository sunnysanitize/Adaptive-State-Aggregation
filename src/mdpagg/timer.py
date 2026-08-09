import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Elapsed:

    wall_ns: int = 0

    @property
    def seconds(self) -> float:
        return self.wall_ns / 1e9


@contextmanager
def timed() -> Iterator[Elapsed]:
    
    elapsed = Elapsed()
    start = time.perf_counter_ns()

    try:
        yield elapsed
    finally:
        elapsed.wall_ns = time.perf_counter_ns() - start
