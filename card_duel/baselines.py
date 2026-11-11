"""Reference players used to calibrate the tasks.

None of them can see the draw pile's order: they use `Battle.blind_copy()`.

Fights:  RandomPlayer < GreedyPlayer < PlannerPlayer
Puzzles: random_line < greedy_line < the exact solver (always solves)
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import ATTACK, CARDS
from .engine import (Battle, end_player_turn, enemy_turn, incoming_damage, play_card, playable_cards,
                     start_player_turn)
from .enemies import Intent
from .fight import FightEnv, FightSpec
from .puzzles import greedy_line, random_line  # re-exported for convenience
from .rng import Rng, mix
from .solver import search_turn

__all__ = ["RandomPlayer", "GreedyPlayer", "PlannerPlayer", "TwoTurnPlanner", "run_fight",
           "FightResult", "candidate_lines", "greedy_line", "random_line"]


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


def candidate_lines(start: Battle, value_fn, limit: int = 5000) -> list[tuple[float, tuple[str, ...], Battle]]:
    """Every distinct end-of-turn position reachable this turn, with its value
    and the shortest line that reaches it, best first."""
    seen: dict[tuple, tuple[float, tuple[str, ...], Battle]] = {}
    stack: list[tuple[Battle, tuple[str, ...]]] = [(start, ())]
    while stack and len(seen) < limit:
        state, line = stack.pop()
        key = state.key()
        if key in seen and len(seen[key][1]) <= len(line):
            continue
        seen[key] = (value_fn(state), line, state)
        if state.outcome() is None:
            for name in playable_cards(state):
                child = state.clone()
                play_card(child, name)
                stack.append((child, line + (name,)))
    return sorted(seen.values(), key=lambda t: (-t[0], len(t[1]), t[1]))


class TwoTurnPlanner:
    """Takes the planner's best few lines and judges each by also planning the
    next turn, averaged over sampled draws. It knows the draw pile's contents
    (shown to every player) but never its order: each sample shuffles it with
    its own seed. Slower, and used to estimate how much headroom fights have."""
    name = "planner (2 turns)"

    def __init__(self, top_k: int = 6, samples: int = 6, seed: int = 0):
        self.top_k, self.samples, self.seed = top_k, samples, seed

    def act(self, env: FightEnv) -> str:
        b = env.battle
        intent, next_intent = env.intent_after(0), env.intent_after(1)
        lines = candidate_lines(b.blind_copy(), plan_value(intent))
        best_line, best_score = (), float("-inf")
        for value, line, end_state in lines[: self.top_k]:
            score = value if abs(value) >= 1e5 else self._next_turn(env, end_state, intent, next_intent)
            if score > best_score + 1e-9:
                best_line, best_score = line, score
        if not best_line:
            return "end turn"
        if "Scout" in best_line:
            return "play Scout"
        return f"play {best_line[0]}"

    def _next_turn(self, env: FightEnv, end_state: Battle, intent: Intent, next_intent: Intent) -> float:
        b = env.battle
        after = end_state.clone()
        end_player_turn(after)
        enemy_turn(after, intent)
        if after.outcome() == "win":
            return 1e6 + after.player.hp
        if after.outcome() == "loss":
            return -1e6
        total = 0.0
        for i in range(self.samples):
            rng = Rng(mix(0x32545552, self.seed, env.spec.seed, b.turn, env.steps, i))
            s = after.clone()
            s.blind = False
            pile = sorted(b.draw_pile)  # sorted first, so the real order can't leak in
            rng.shuffle(pile)
            drawn_now = pile[: s.pending_draws]  # cards Scout drew this turn, now discarded
            s.draw_pile = pile[s.pending_draws:]
            s.discard = sorted(b.discard) + list(after.discard) + drawn_now
            s.pending_draws = 0
            s.seed = rng.next_u64()  # a reshuffle must not reveal the real order
            start_player_turn(s)
            if s.outcome() == "loss":
                total += -1e6
                continue
            total += search_turn(s.blind_copy(), plan_value(next_intent)).value
        return total / self.samples


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
