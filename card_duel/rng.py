"""Deterministic random numbers (SplitMix64).

Python's `random` module does not promise identical sequences across
versions, so every seeded task in this package uses this generator instead.
The same seed gives the same puzzle or fight on any machine and any Python.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

MASK64 = (1 << 64) - 1
T = TypeVar("T")


class Rng:
    def __init__(self, seed: int) -> None:
        self._state = seed & MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Uniform integer in [0, n) without modulo bias."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.next_u64()
            if x < limit:
                return x % n

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi], both ends included."""
        return lo + self.below(hi - lo + 1)

    def choice(self, items: Sequence[T]) -> T:
        return items[self.below(len(items))]

    def weighted(self, items: Sequence[tuple[T, int]]) -> T:
        total = sum(w for _, w in items)
        pick = self.below(total)
        for item, weight in items:
            if pick < weight:
                return item
            pick -= weight
        raise AssertionError("unreachable")

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]


def mix(*parts: int) -> int:
    """Combine integers into one well-spread 64-bit seed."""
    h = 0x243F6A8885A308D3
    for part in parts:
        h = Rng(h ^ (part & MASK64)).next_u64()
    return h
