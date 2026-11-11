"""Reference players used to calibrate the tasks.

None of them can see the draw pile's order: they use `Battle.blind_copy()`.

Fights:  RandomPlayer < GreedyPlayer < PlannerPlayer
Puzzles: random_line < greedy_line < the exact solver (always solves)
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import ATTACK, CARDS
from .engine import Battle, end_player_turn, enemy_turn, incoming_damage, play_card, playable_cards
from .enemies import Intent
from .fight import FightEnv, FightSpec
from .puzzles import greedy_line, random_line  # re-exported for convenience
from .rng import Rng, mix
from .solver import search_turn

__all__ = ["RandomPlayer", "GreedyPlayer", "PlannerPlayer", "run_fight", "FightResult",
           "greedy_line", "random_line"]


class RandomPlayer:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = Rng(mix(0x52414E44, seed))

    def act(self, env: FightEnv) -> str:
        return self.rng.choice(env.legal_actions())


class GreedyPlayer:
    """Plays the card with the best immediate payoff per energy, blocking big hits."""
    name = "greedy"

    def act(self, env: FightEnv) -> str:
        b = env.battle
        need = max(0, incoming_damage(b, env.intent) - b.player.shield)
        best, best_score = None, 0.0
        for name in playable_cards(b):
            c = b.blind_copy()
            play_card(c, name)
            if c.outcome() == "win":
                return f"play {name}"
            if c.outcome() == "loss":
                continue
            score = (b.enemy.hp - c.enemy.hp) + min(c.player.shield - b.player.shield, need)
            score += 1.5 * (c.enemy.burn - b.enemy.burn) + 2.0 * (c.enemy.cracked - b.enemy.cracked)
            score += {"Hone": 3.0, "Scout": 2.0}.get(name, 0.0)
            if name == "Wind Up" and any(CARDS[n].kind == ATTACK and CARDS[n].cost <= c.energy for n in c.hand):
                score += 2.0
            if name == "Overclock":
                score += 2.0 if b.player.hp > 20 else -10.0
            cost = CARDS[name].cost
            score /= cost if cost > 0 else 0.5
            if score > best_score + 1e-9:
                best, best_score = name, score
        return f"play {best}" if best else "end turn"


def plan_value(intent: Intent):
    """Score the position after the coming enemy turn."""
    def value(s: Battle) -> float:
        if s.player.hp <= 0:
            return -1e6
        if s.enemy.hp <= 0:
            return 1e6 + s.player.hp
        c = s.clone()
        end_player_turn(c)
        enemy_turn(c, intent)
        if c.enemy.hp <= 0:
            return 1e6 + c.player.hp
        if c.player.hp <= 0:
            return -1e6 - c.enemy.hp
        return (c.player.hp - c.enemy.hp
                + 1.5 * c.enemy.burn + 2.0 * c.enemy.cracked + 3.0 * c.player.might
                - 1.5 * c.player.burn - 2.0 * c.player.cracked + 2.0 * s.pending_draws)
    return value


class PlannerPlayer:
    """Searches every line for this turn, scores it after the enemy's reply,
    plays the first card of the best line, then plans again."""
    name = "planner"

    def act(self, env: FightEnv) -> str:
        res = search_turn(env.battle.blind_copy(), plan_value(env.intent))
        if not res.line:
            return "end turn"
        if "Scout" in res.line:
            return "play Scout"  # draw first: it costs nothing and reveals cards
        return f"play {res.line[0]}"


@dataclass(frozen=True)
class FightResult:
    seed: int
    won: bool
    reward: float
    turns: int
    hp_left: int
    illegal: int


def run_fight(spec: FightSpec | int, player, record: list[str] | None = None) -> FightResult:
    env = FightEnv(spec)
    obs = env.intro()
    if record is not None:
        record.append(obs)
    info: dict = {}
    while not env.done:
        action = player.act(env)
        obs, _, _, info = env.step(action)
        if record is not None:
            record.append(f">>> {action}")
            record.append(obs)
    return FightResult(env.spec.seed, env.won, env.reward(), env.battle.turn,
                       max(env.battle.player.hp, 0), env.illegal)
