"""Two-turn planning puzzles: "win by the end of your second turn".

The draw pile's order is shown, the enemy's one action between your turns is
shown, and no reshuffle can happen, so everything is known. An exact nested
search finds the most damage any pair of turns can deal while surviving the
enemy's action; the enemy's HP is set to exactly that. Only the best plan
wins, and it often means giving up damage now (for Shield, Burn or set-up)
to win later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterator

from .cards import CARDS
from .enemies import ENEMIES, ENEMY_NAMES, Intent, describe_intent
from .engine import (DRAW_PER_TURN, Battle, Fighter, card_error, end_player_turn, enemy_turn,
                     incoming_damage, play_card, playable_cards, start_player_turn)
from .puzzles import NON_SCOUT, PLAYER_MAX_HP, POOL, damage_value, finish_turn, greedy_line
from .rng import Rng, mix
from .solver import search_turn
from .text import RULES, card_reference, extract_tag, list_cards, parse_line

PLAN_STREAM = 0x504C414E
TEST_EVERY = 5
DEAD = -1e9


@dataclass(frozen=True)
class PlanPuzzle:
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
    intent: tuple[tuple, ...]
    solution: tuple[tuple[str, ...], tuple[str, ...]]

    def battle(self, enemy_hp: int | None = None) -> Battle:
        hp = self.enemy_hp if enemy_hp is None else enemy_hp
        return Battle(
            player=Fighter("You", self.player_hp, PLAYER_MAX_HP, might=self.might),
            enemy=Fighter(self.enemy_name, hp, max(hp, self.enemy_max_hp), shield=self.enemy_shield,
                          burn=self.enemy_burn, cracked=self.enemy_cracked),
            energy=self.energy, max_energy=self.energy, hand=list(self.hand),
            draw_pile=list(self.draw_pile), seed=mix(PLAN_STREAM, self.seed, 1),
        )

    def prompt(self) -> str:
        return render_plan_puzzle(self)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, d: dict) -> "PlanPuzzle":
        d = dict(d)
        d["hand"], d["draw_pile"] = tuple(d["hand"]), tuple(d["draw_pile"])
        d["intent"] = tuple(tuple(e) for e in d["intent"])
        d["solution"] = (tuple(d["solution"][0]), tuple(d["solution"][1]))
        return cls(**d)


@dataclass(frozen=True)
class PlanResult:
    reward: float
    solved: bool
    legal: bool
    damage: int
    played: tuple[tuple[str, ...], tuple[str, ...]]
    error: str | None


# ----------------------------------------------------------------- rules glue

def between_turns(b: Battle, intent: Intent) -> None:
    """End turn 1, run the enemy's action, start turn 2 (unless the fight is over)."""
    if b.outcome() is not None:
        return
    end_player_turn(b)
    enemy_turn(b, intent)
    if b.outcome() is None:
        start_player_turn(b)


def _second_turn_value(start_hp: int, intent: Intent, cap: int | None):
    turn2 = damage_value(start_hp, cap)

    def value(s: Battle) -> float:
        if s.player.hp <= 0:
            return DEAD
        c = s.clone()
        between_turns(c, intent)
        if c.outcome() == "loss":
            return DEAD
        if c.outcome() == "win":
            lost = start_hp - c.enemy.hp
            return float(lost if cap is None else min(lost, cap))
        return search_turn(c, turn2).value
    return value


def solve(b: Battle, intent: Intent, cap: int | None = None) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """Best total damage over two turns, with the plan that reaches it."""
    start_hp = b.enemy.hp
    first = search_turn(b, _second_turn_value(start_hp, intent, cap))
    if first.value <= DEAD / 2:
        return first.value, first.line, ()
    after = b.clone()
    for name in first.line:
        play_card(after, name)
    between_turns(after, intent)
    if after.outcome() is not None:
        return first.value, first.line, ()
    second = search_turn(after, damage_value(start_hp, cap))
    return first.value, first.line, second.line


# ----------------------------------------------------------------- generation

def _candidate(rng: Rng) -> dict:
    hand = [rng.weighted(POOL) for _ in range(rng.randint(4, 5))]
    draw = [rng.weighted(POOL) for _ in range(rng.randint(5, 7))]
    scouts = hand.count("Scout") + draw.count("Scout")
    while len(draw) < DRAW_PER_TURN + 2 * scouts:  # every draw is known: no reshuffle
        draw.append(rng.weighted(NON_SCOUT))
    enemy = rng.choice(ENEMY_NAMES)
    intent = rng.choice(ENEMIES[enemy].pattern)
    return {
        "hand": tuple(hand), "draw_pile": tuple(draw), "energy": rng.choice((2, 3, 3, 3)),
        "might": rng.choice((0, 0, 0, 1)), "player_hp": rng.randint(6, 30), "enemy_name": enemy,
        "enemy_shield": rng.choice((0, 0, 0, 4, 6, 10)), "enemy_burn": rng.choice((0, 0, 0, 2, 3)),
        "enemy_cracked": rng.choice((0, 0, 0, 1)), "intent": intent,
    }


def _draft(seed: int, c: dict, enemy_hp: int, solution=((), ())) -> PlanPuzzle:
    return PlanPuzzle(
        id=f"plan-{seed}", seed=seed, split="", difficulty="", player_hp=c["player_hp"],
        energy=c["energy"], might=c["might"], hand=c["hand"], draw_pile=c["draw_pile"],
        enemy_name=c["enemy_name"], enemy_hp=enemy_hp,
        enemy_max_hp=max(enemy_hp, ENEMIES[c["enemy_name"]].max_hp), enemy_shield=c["enemy_shield"],
        enemy_burn=c["enemy_burn"], enemy_cracked=c["enemy_cracked"], intent=c["intent"],
        solution=solution,
    )


def _split_of(c: dict) -> str:
    sig = json.dumps([sorted(c["hand"]), c["draw_pile"], c["energy"], c["might"], c["player_hp"],
                      c["enemy_hp"], c["enemy_shield"], c["enemy_burn"], c["enemy_cracked"],
                      c["intent"]])
    return "test" if int(hashlib.sha256(sig.encode()).hexdigest()[:8], 16) % TEST_EVERY == 0 else "train"


def generate(seed: int) -> PlanPuzzle:
    rng = Rng(mix(PLAN_STREAM, seed))
    while True:
        c = _candidate(rng)
        probe = _draft(seed, c, 10**6)
        value, _, _ = solve(probe.battle(), probe.intent)
        target = int(value)
        if not 18 <= target <= 120:
            continue  # unsurvivable, or too little damage to be interesting
        draft = _draft(seed, c, target)
        value, t1, t2 = solve(draft.battle(), draft.intent, cap=target)
        assert int(value) == target, "a winning plan must exist"
        if not t2:
            continue  # winnable in one turn: that's the other puzzle type
        c["enemy_hp"] = target
        draft = _draft(seed, c, target, (t1, t2))
        # "medium": dealing the most damage now, then the best second turn, is enough.
        # "hard": the winning plan gives up damage in turn 1 to win in turn 2.
        difficulty = "medium" if score_plan(draft, *myopic_plan(draft)).solved else "hard"
        return PlanPuzzle(**{**asdict(draft), "split": _split_of(c), "difficulty": difficulty})


def iter_plan_puzzles(split: str | None = None, difficulty: str | None = None,
                      start_seed: int = 0) -> Iterator[PlanPuzzle]:
    seed = start_seed
    while True:
        p = generate(seed)
        seed += 1
        if (split is None or p.split == split) and (difficulty is None or p.difficulty == difficulty):
            yield p


def plan_dataset(split: str = "train", n: int = 100, difficulty: str | None = None,
                 start_seed: int = 0) -> list[PlanPuzzle]:
    out = []
    for p in iter_plan_puzzles(split, difficulty, start_seed):
        out.append(p)
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------- scoring

def score_plan(p: PlanPuzzle, turn1, turn2, partial_credit: bool = False) -> PlanResult:
    b = p.battle()
    played: list[list[str]] = [[], []]
    for turn, names in enumerate((turn1, turn2)):
        if turn == 1:
            between_turns(b, p.intent)
        for name in names:
            if b.outcome() is not None:
                break
            err = card_error(b, name)
            if err:
                return PlanResult(0.0, False, False, p.enemy_hp - b.enemy.hp,
                                  (tuple(played[0]), tuple(played[1])),
                                  f"illegal play in turn {turn + 1}: {err}")
            play_card(b, name)
            played[turn].append(name)
        if b.outcome() is not None:
            break
    if b.outcome() is None:
        finish_turn(b)
    done = (tuple(played[0]), tuple(played[1]))
    damage = min(p.enemy_hp, p.enemy_hp - b.enemy.hp)
    if b.player.hp <= 0:
        return PlanResult(0.0, False, True, damage, done, "your HP reached 0")
    solved = b.enemy.hp <= 0
    reward = 1.0 if solved else (0.5 * damage / p.enemy_hp if partial_credit else 0.0)
    return PlanResult(reward, solved, True, damage, done, None)


def score_plan_answer(p: PlanPuzzle, completion: str, partial_credit: bool = False) -> PlanResult:
    """Read "<answer>turn 1 cards | turn 2 cards</answer>" and score it."""
    answer = extract_tag(completion, "answer")
    if answer is None:
        return PlanResult(0.0, False, False, 0, ((), ()), "no <answer>...</answer> found")
    parts = answer.split("|")
    if len(parts) > 2:
        return PlanResult(0.0, False, False, 0, ((), ()), "use one '|' between turn 1 and turn 2")
    lines = []
    for part in parts + [""] * (2 - len(parts)):
        names, err = parse_line(part)
        if err:
            return PlanResult(0.0, False, False, 0, ((), ()), err)
        lines.append(names)
    return score_plan(p, lines[0], lines[1], partial_credit)


# ----------------------------------------------------------------- simple players

def greedy_plan(p: PlanPuzzle) -> tuple[list[str], list[str]]:
    """Greedy damage each turn, ignoring the enemy's action."""
    b = p.battle()
    t1 = greedy_line(_as_one_turn(p, b))
    for name in t1:
        play_card(b, name)
    between_turns(b, p.intent)
    if b.outcome() is not None:
        return t1, []
    return t1, _greedy_from(b, p.enemy_hp)


def myopic_plan(p: PlanPuzzle) -> tuple[list[str], list[str]]:
    """The most damage possible now (while surviving), then the best second turn."""
    b = p.battle()
    start_hp = b.enemy.hp

    def now_value(s: Battle) -> float:
        if s.player.hp <= 0:
            return DEAD
        c = s.clone()
        between_turns(c, p.intent)
        if c.outcome() == "loss":
            return DEAD
        return float(start_hp - s.enemy.hp) + 0.001 * (c.player.hp if c.outcome() is None else 0)

    t1 = list(search_turn(b, now_value).line)
    for name in t1:
        play_card(b, name)
    between_turns(b, p.intent)
    if b.outcome() is not None:
        return t1, []
    return t1, list(search_turn(b, damage_value(start_hp)).line)


def _greedy_from(b: Battle, start_hp: int) -> list[str]:
    value = damage_value(start_hp)
    line: list[str] = []
    while b.outcome() is None:
        now = value(b)
        best, gain_best = None, 0.0
        for name in playable_cards(b):
            c = b.clone()
            play_card(c, name)
            cost = CARDS[name].cost
            gain = (value(c) - now) / (cost if cost > 0 else 0.5)
            if gain > gain_best + 1e-9:
                best, gain_best = name, gain
        if best is None:
            break
        play_card(b, best)
        line.append(best)
    return line


def _as_one_turn(p: PlanPuzzle, b: Battle):
    from .puzzles import Puzzle
    return Puzzle(id=p.id, seed=p.seed, split=p.split, difficulty="", player_hp=p.player_hp,
                  energy=p.energy, might=p.might, hand=p.hand, draw_pile=p.draw_pile,
                  enemy_name=p.enemy_name, enemy_hp=p.enemy_hp, enemy_max_hp=p.enemy_max_hp,
                  enemy_shield=p.enemy_shield, enemy_burn=p.enemy_burn, enemy_cracked=p.enemy_cracked,
                  solution=(), winning_lines=0, total_lines=0)


# ----------------------------------------------------------------- prompt

def render_plan_puzzle(p: PlanPuzzle) -> str:
    cards = list(p.hand) + list(p.draw_pile)
    probe = p.battle()
    incoming = incoming_damage(probe, p.intent)
    return f"""You are playing Card Duel, a turn-based card game.

{RULES}

PUZZLE: WIN WITHIN TWO TURNS
Plan your next two turns so the enemy is defeated before its second action.
- Turn 1: play cards from your hand. Then your turn ends (your hand is discarded) and the enemy takes ONE action: {describe_intent(p.intent)} ({incoming} damage to you before Shield, as things stand now).
- Turn 2: you lose your Shield, your Burn ticks, energy refills to {p.energy} and you draw 5 cards from the top of the draw pile. Play cards, then your turn ends.
- You win if the enemy's HP reaches 0 at any point up to the start of its next turn (when its Burn ticks again). You lose if your HP reaches 0.
- The draw pile's order is shown, and it is long enough that it will not run out.

STATE
You: HP {p.player_hp}/{PLAYER_MAX_HP}, Energy {p.energy}, Might {p.might}, Charge 0
Enemy ({p.enemy_name}): HP {p.enemy_hp}/{p.enemy_max_hp}, Shield {p.enemy_shield}, Burn {p.enemy_burn}, Cracked {p.enemy_cracked}, Might 0
Hand: {list_cards(p.hand)}
Draw pile, top card first: {list_cards(p.draw_pile, keep_order=True)}
Discard pile: empty

CARDS
{card_reference(cards)}

Answer with the cards for turn 1, then a "|", then the cards for turn 2, all inside <answer></answer>.
Example: <answer>Guard, Jab | Wind Up, Flurry, Jab</answer>"""
