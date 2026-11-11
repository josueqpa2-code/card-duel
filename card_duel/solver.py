"""Exhaustive search over every way to play out one turn.

`search_turn` tries every order of card plays (identical cards are merged)
and scores each finished line with a value function. It returns the best
value, the best line (shortest, then alphabetical, among ties), how many
lines reach the best value and how many distinct lines exist in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .engine import Battle, card_error, play_card

EPS = 1e-9


@dataclass(frozen=True)
class SearchResult:
    value: float
    line: tuple[str, ...]
    best_lines: int
    total_lines: int


def search_turn(start: Battle, value_fn: Callable[[Battle], float]) -> SearchResult:
    memo: dict[tuple, tuple[float, tuple[str, ...], int, int]] = {}

    def rec(state: Battle) -> tuple[float, tuple[str, ...], int, int]:
        key = state.key()
        hit = memo.get(key)
        if hit is not None:
            return hit
        best_value = value_fn(state)
        best_line: tuple[str, ...] = ()
        n_best, n_total = 1, 1  # ending the turn now is one line
        if state.outcome() is None:
            for name in sorted(set(state.hand)):
                if card_error(state, name) is not None:
                    continue
                child = state.clone()
                play_card(child, name)
                c_value, c_line, c_best, c_total = rec(child)
                line = (name,) + c_line
                n_total += c_total
                if c_value > best_value + EPS:
                    best_value, best_line, n_best = c_value, line, c_best
                elif abs(c_value - best_value) <= EPS:
                    n_best += c_best
                    if (len(line), line) < (len(best_line), best_line):
                        best_line = line
        result = (best_value, best_line, n_best, n_total)
        memo[key] = result
        return result

    value, line, n_best, n_total = rec(start)
    return SearchResult(value, line, n_best, n_total)
