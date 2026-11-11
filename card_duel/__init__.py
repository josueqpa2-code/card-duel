"""Card Duel: an original turn-based card game packaged as an AI training and
evaluation environment. Pure Python standard library; no dependencies.

Two task types:
- puzzles: single-turn "win this turn" problems with a provably optimal answer
- fights: full multi-turn battles scored on winning and HP kept
"""

from .puzzles import Puzzle, PuzzleResult, dataset, generate, score_answer, score_line
from .fight import FightEnv, FightSpec, fight_dataset, make_fight

__version__ = "0.1.0"

__all__ = [
    "Puzzle", "PuzzleResult", "dataset", "generate", "score_answer", "score_line",
    "FightEnv", "FightSpec", "fight_dataset", "make_fight", "load_environment",
]


def load_environment(**kwargs):
    """Build a `verifiers` environment (needs `pip install verifiers datasets`)."""
    from .verifiers_env import load_environment as _load
    return _load(**kwargs)
