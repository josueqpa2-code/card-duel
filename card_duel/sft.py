"""Supervised fine-tuning data built from the exact solver and the planners.

Puzzle examples end with a worked solution: every step is computed by the
engine, so the reasoning is correct by construction, followed by the
<answer> tag the scorer reads. Fight examples are full conversations played
by a reference player, one action per assistant message.
"""

from __future__ import annotations

from .baselines import PlannerPlayer, TwoTurnPlanner
from .cards import CARDS
from .engine import play_card
from .fight import FightEnv, FightSpec
from .puzzles import Puzzle, finish_turn
from .verifiers_env import FIGHT_SYSTEM, PUZZLE_SYSTEM


def puzzle_explanation(p: Puzzle) -> str:
    """Step-by-step account of the solution line, ending in the answer tag."""
    b = p.battle()
    lines = [f"The enemy has {p.enemy_hp} HP and {p.enemy_shield} Shield. I have {p.energy} energy."]
    for i, name in enumerate(p.solution, 1):
        hp, shield, energy = b.enemy.hp, b.enemy.shield, b.energy
        burn, charge, might = b.enemy.burn, b.charge, b.player.might
        hand_before = len(b.hand)
        play_card(b, name)
        bits = []
        if b.enemy.hp < hp:
            bits.append(f"deals {hp - b.enemy.hp} damage to HP")
        if b.enemy.shield < shield:
            bits.append(f"removes {shield - b.enemy.shield} Shield")
        if b.enemy.burn != burn:
            bits.append(f"Burn {burn} -> {b.enemy.burn}")
        if b.charge > charge:
            bits.append(f"Charge {b.charge}")
        if b.player.might > might:
            bits.append(f"Might {b.player.might}")
        if len(b.hand) > hand_before - 1:
            drawn = len(b.hand) - (hand_before - 1)
            bits.append(f"draws {', '.join(b.hand[-drawn:])}")
        if b.energy > energy - CARDS[name].cost:
            bits.append(f"energy {b.energy}")
        detail = ", ".join(bits) if bits else "no immediate effect"
        lines.append(f"{i}. {name} ({CARDS[name].cost} energy): {detail}. "
                     f"Enemy at {max(b.enemy.hp, 0)} HP, {b.enemy.shield} Shield; {b.energy} energy left.")
        if b.enemy.hp <= 0:
            break
    if b.enemy.hp > 0:
        before = b.enemy.hp
        finish_turn(b)
        lines.append(f"At the start of the enemy's turn its Burn deals {before - b.enemy.hp} damage, "
                     f"bringing it to {max(b.enemy.hp, 0)} HP.")
    lines.append("The enemy is defeated before it can act.")
    lines.append(f"<answer>{', '.join(p.solution)}</answer>")
    return "\n".join(lines)


def puzzle_example(p: Puzzle) -> dict:
    return {"id": p.id, "task": "puzzle", "difficulty": p.difficulty, "messages": [
        {"role": "system", "content": PUZZLE_SYSTEM},
        {"role": "user", "content": p.prompt()},
        {"role": "assistant", "content": puzzle_explanation(p)},
    ]}


def fight_example(spec: FightSpec, player=None) -> dict:
    """A full fight played by a reference player (default: the two-turn planner)."""
    player = player or TwoTurnPlanner()
    env = FightEnv(spec)
    messages = [{"role": "system", "content": FIGHT_SYSTEM}, {"role": "user", "content": env.intro()}]
    while not env.done:
        action = player.act(env)
        obs, *_ = env.step(action)
        messages.append({"role": "assistant", "content": f"<action>{action}</action>"})
        if not env.done:
            messages.append({"role": "user", "content": obs})
    return {"id": spec.id, "task": "fight", "won": env.won, "reward": env.reward(),
            "player": getattr(player, "name", type(player).__name__), "messages": messages}


__all__ = ["puzzle_explanation", "puzzle_example", "fight_example", "PlannerPlayer", "TwoTurnPlanner"]
