"""Single-turn "win this turn" puzzles.

Each puzzle is a seeded game state. The enemy's HP is set to exactly the
most damage any line of play can deal this turn (found by exhaustive search),
so a solution always exists and only best-possible play wins.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterator

from .cards import CARDS
from .enemies import ENEMIES, ENEMY_NAMES
from .engine import Battle, Fighter, card_error, end_player_turn, play_card, playable_cards, start_of_turn
from .rng import Rng, mix
from .solver import search_turn
from .text import RULES, card_reference, extract_tag, list_cards, parse_line

PUZZLE_STREAM = 0x50555A5A
TEST_EVERY = 5  # about 1 in 5 puzzles belongs to the test split
PLAYER_MAX_HP = 40
MIN_DAMAGE = 10
MAX_DAMAGE = 90
PROBE_HP = 10**6

POOL = [
    ("Jab", 6), ("Flurry", 4), ("Crack", 3), ("Finisher", 3), ("Cinder Strike", 3),
    ("Shatter", 2), ("Spark", 3), ("Guard", 3), ("Bulwark", 1), ("Ignite", 3),
    ("Stoke", 2), ("Wind Up", 3), ("Scout", 2), ("Overclock", 2), ("Hone", 1),
]
NON_SCOUT = [(n, w) for n, w in POOL if n != "Scout"]


@dataclass(frozen=True)
class Puzzle:
    id: str
    seed: int
    split: str
    difficulty: str
    player_hp: int
    energy: int
    might: int
    hand: tuple[str, ...]
    draw_pile: tuple[str, ...]
    enemy_name: str
    enemy_hp: int
    enemy_max_hp: int
    enemy_shield: int
    enemy_burn: int
    enemy_cracked: int
    solution: tuple[str, ...]
    winning_lines: int
    total_lines: int

    def battle(self, enemy_hp: int | None = None) -> Battle:
        hp = self.enemy_hp if enemy_hp is None else enemy_hp
        return Battle(
            player=Fighter("You", self.player_hp, PLAYER_MAX_HP, might=self.might),
            enemy=Fighter(self.enemy_name, hp, max(hp, self.enemy_max_hp),
                          shield=self.enemy_shield, burn=self.enemy_burn,
                          cracked=self.enemy_cracked),
            energy=self.energy,
            max_energy=self.energy,
            hand=list(self.hand),
            draw_pile=list(self.draw_pile),
            seed=mix(PUZZLE_STREAM, self.seed, 1),
        )

    def prompt(self) -> str:
        return render_puzzle(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Puzzle":
        d = dict(d)
        for k in ("hand", "draw_pile", "solution"):
            d[k] = tuple(d[k])
        return cls(**d)


@dataclass(frozen=True)
class PuzzleResult:
    reward: float
    solved: bool
    legal: bool
    damage: int
    played: tuple[str, ...]
    error: str | None


# ----------------------------------------------------------------- rules glue

def finish_turn(b: Battle) -> None:
    """End the player's turn and run the enemy's start-of-turn upkeep."""
    if b.outcome() is not None:
        return
    end_player_turn(b)
    start_of_turn(b.enemy)


def damage_value(start_hp: int, cap: int | None = None):
    """Value of a line = enemy HP lost by the start of its turn (own death = -inf)."""
    def value(b: Battle) -> float:
        if b.player.hp <= 0:
            return -1e9
        c = b.clone()
        finish_turn(c)
        lost = start_hp - c.enemy.hp
        return float(lost if cap is None else min(lost, cap))
    return value


# ----------------------------------------------------------------- generation

def _signature(c: dict) -> str:
    return json.dumps([sorted(c["hand"]), c["draw_pile"], c["energy"], c["might"], c["player_hp"],
                       c["enemy_hp"], c["enemy_shield"], c["enemy_burn"], c["enemy_cracked"]])


def _split_of(signature: str) -> str:
    digest = int(hashlib.sha256(signature.encode()).hexdigest()[:8], 16)
    return "test" if digest % TEST_EVERY == 0 else "train"


def _candidate(rng: Rng) -> dict:
    hand = [rng.weighted(POOL) for _ in range(rng.randint(4, 7))]
    draw = [rng.weighted(POOL) for _ in range(rng.randint(0, 3))]
    scouts = hand.count("Scout") + draw.count("Scout")
    while len(draw) < 2 * scouts:  # every draw is known: no reshuffle can happen
        draw.append(rng.weighted(NON_SCOUT))
    return {
        "hand": tuple(hand),
        "draw_pile": tuple(draw),
        "energy": rng.choice((2, 3, 3, 3, 4)),
        "might": rng.choice((0, 0, 0, 0, 1, 2)),
        "player_hp": rng.randint(3, 30),
        "enemy_name": rng.choice(ENEMY_NAMES),
        "enemy_shield": rng.choice((0, 0, 0, 4, 6, 10)),
        "enemy_burn": rng.choice((0, 0, 0, 2, 4)),
        "enemy_cracked": rng.choice((0, 0, 0, 1)),
    }


def _draft(seed: int, c: dict, enemy_hp: int) -> Puzzle:
    return Puzzle(
        id=f"puzzle-{seed}", seed=seed, split="", difficulty="",
        player_hp=c["player_hp"], energy=c["energy"], might=c["might"],
        hand=c["hand"], draw_pile=c["draw_pile"], enemy_name=c["enemy_name"],
        enemy_hp=enemy_hp, enemy_max_hp=max(enemy_hp, ENEMIES[c["enemy_name"]].max_hp),
        enemy_shield=c["enemy_shield"], enemy_burn=c["enemy_burn"],
        enemy_cracked=c["enemy_cracked"], solution=(), winning_lines=0, total_lines=0,
    )


def generate(seed: int) -> Puzzle:
    """The puzzle for a seed. Deterministic on every machine."""
    rng = Rng(mix(PUZZLE_STREAM, seed))
    while True:
        c = _candidate(rng)
        probe = _draft(seed, c, PROBE_HP)
        best = search_turn(probe.battle(), damage_value(PROBE_HP))
        target = int(best.value)
        if not MIN_DAMAGE <= target <= MAX_DAMAGE:
            continue
        draft = _draft(seed, c, target)
        res = search_turn(draft.battle(), damage_value(target, cap=target))
        assert int(res.value) == target, "a winning line must exist"
        if len(res.line) < 2 or res.best_lines * 2 > res.total_lines:
            continue  # one-card wins and puzzles most lines solve are too easy
        c["enemy_hp"] = target
        draft = _draft(seed, c, target)
        draft = Puzzle(**{**draft.to_dict(), "solution": res.line,
                          "winning_lines": res.best_lines, "total_lines": res.total_lines})
        greedy_wins = score_line(draft, greedy_line(draft)).solved
        share = res.best_lines / res.total_lines
        difficulty = "easy" if greedy_wins else ("medium" if share >= 0.02 else "hard")
        return Puzzle(**{**draft.to_dict(), "split": _split_of(_signature(c)),
                         "difficulty": difficulty})


def iter_puzzles(split: str | None = None, difficulty: str | None = None,
                 start_seed: int = 0) -> Iterator[Puzzle]:
    seed = start_seed
    while True:
        p = generate(seed)
        seed += 1
        if (split is None or p.split == split) and (difficulty is None or p.difficulty == difficulty):
            yield p


def dataset(split: str = "train", n: int = 100, difficulty: str | None = None,
            start_seed: int = 0) -> list[Puzzle]:
    out = []
    for p in iter_puzzles(split, difficulty, start_seed):
        out.append(p)
        if len(out) >= n:
            return out
    return out


# ----------------------------------------------------------------- scoring

def score_line(p: Puzzle, names: list[str] | tuple[str, ...], partial_credit: bool = False) -> PuzzleResult:
    b = p.battle()
    played: list[str] = []
    for name in names:
        if b.outcome() is not None:
            break  # cards after the win (or your own defeat) don't matter
        err = card_error(b, name)
        if err:
            return PuzzleResult(0.0, False, False, p.enemy_hp - b.enemy.hp, tuple(played),
                                f"illegal play #{len(played) + 1}: {err}")
        play_card(b, name)
        played.append(name)
    if b.player.hp <= 0:
        return PuzzleResult(0.0, False, True, p.enemy_hp - b.enemy.hp, tuple(played),
                            "your HP reached 0")
    finish_turn(b)
    damage = min(p.enemy_hp, p.enemy_hp - b.enemy.hp)
    solved = b.enemy.hp <= 0
    reward = 1.0 if solved else (0.5 * damage / p.enemy_hp if partial_credit else 0.0)
    return PuzzleResult(reward, solved, True, damage, tuple(played), None)


def score_answer(p: Puzzle, completion: str, partial_credit: bool = False) -> PuzzleResult:
    """Score a model reply. The play order is read from the last <answer> tag."""
    answer = extract_tag(completion, "answer")
    if answer is None:
        names, err = parse_line(completion)
        if err or not names:
            return PuzzleResult(0.0, False, False, 0, (), "no <answer>...</answer> found")
    else:
        names, err = parse_line(answer)
        if err:
            return PuzzleResult(0.0, False, False, 0, (), err)
    return score_line(p, names, partial_credit)


# ----------------------------------------------------------------- simple players

def greedy_line(p: Puzzle) -> list[str]:
    """Always play the card that adds the most damage right now, per energy."""
    b = p.battle()
    value = damage_value(p.enemy_hp)
    line: list[str] = []
    while b.outcome() is None:
        now = value(b)
        best_name, best_gain = None, 0.0
        for name in playable_cards(b):
            c = b.clone()
            play_card(c, name)
            cost = CARDS[name].cost
            gain = (value(c) - now) / (cost if cost > 0 else 0.5)
            if gain > best_gain + 1e-9:
                best_name, best_gain = name, gain
        if best_name is None:
            break
        play_card(b, best_name)
        line.append(best_name)
    return line


def _best_within(b: Battle, depth: int, value) -> float:
    best = value(b)
    if depth == 0 or b.outcome() is not None:
        return best
    for name in playable_cards(b):
        c = b.clone()
        play_card(c, name)
        best = max(best, _best_within(c, depth - 1, value))
    return best


def lookahead_line(p: Puzzle, depth: int = 2) -> list[str]:
    """Like greedy, but each choice looks `depth` cards ahead before committing."""
    b = p.battle()
    value = damage_value(p.enemy_hp)
    line: list[str] = []
    while b.outcome() is None:
        best_name, best_value = None, value(b)
        for name in playable_cards(b):
            c = b.clone()
            play_card(c, name)
            v = _best_within(c, depth - 1, value)
            if v > best_value + 1e-9:
                best_name, best_value = name, v
        if best_name is None:
            break
        play_card(b, best_name)
        line.append(best_name)
    return line


def random_line(p: Puzzle, seed: int) -> list[str]:
    """Play random affordable cards, ending the turn at a random point."""
    rng = Rng(mix(PUZZLE_STREAM, p.seed, seed, 7))
    b = p.battle()
    line: list[str] = []
    while b.outcome() is None:
        options = playable_cards(b)
        if not options or rng.below(len(options) + 1) == len(options):
            break
        name = rng.choice(options)
        play_card(b, name)
        line.append(name)
    return line


# ----------------------------------------------------------------- prompt

def render_puzzle(p: Puzzle) -> str:
    cards = list(p.hand) + list(p.draw_pile)
    return f"""You are playing Card Duel, a turn-based card game.

{RULES}

PUZZLE: WIN THIS TURN
Choose which cards to play this turn, and in what order, so the enemy is defeated before it can act.
- When you stop playing cards your turn ends. The enemy's turn then starts: it loses its Shield and its Burn ticks. If that brings its HP to 0, you win.
- There are no later turns. If the enemy is still alive after its Burn ticks, you lose.
- If your own HP reaches 0, you lose.

STATE
You: HP {p.player_hp}/{PLAYER_MAX_HP}, Energy {p.energy}, Might {p.might}, Charge 0
Enemy ({p.enemy_name}): HP {p.enemy_hp}/{p.enemy_max_hp}, Shield {p.enemy_shield}, Burn {p.enemy_burn}, Cracked {p.enemy_cracked}
Hand: {list_cards(p.hand)}
Draw pile, top card first: {list_cards(p.draw_pile, keep_order=True)}
Discard pile: empty

CARDS
{card_reference(cards)}

Answer with the cards to play, in order, separated by commas, inside <answer></answer>.
Example: <answer>Wind Up, Jab, Jab</answer>"""
