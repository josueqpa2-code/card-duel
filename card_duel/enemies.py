"""Enemies and their intents.

An intent is a tuple of effects done in order:
    ("attack", damage, hits)  ("shield", n)  ("burn", n)  ("crack", n)  ("might", n)
Each enemy repeats its pattern in a fixed cycle, so its next move is always
visible to the player.
"""

from __future__ import annotations

from dataclasses import dataclass

Effect = tuple
Intent = tuple[Effect, ...]


@dataclass(frozen=True)
class EnemyKind:
    name: str
    max_hp: int
    start_shield: int
    pattern: tuple[Intent, ...]


ENEMIES: dict[str, EnemyKind] = {
    e.name: e
    for e in [
        EnemyKind("Bramble Golem", 80, 0, (
            (("attack", 12, 1),),
            (("shield", 10), ("attack", 7, 1)),
            (("attack", 20, 1),),
        )),
        EnemyKind("Ash Wisp", 64, 0, (
            (("burn", 5), ("attack", 7, 1)),
            (("attack", 6, 2),),
            (("might", 3), ("shield", 10)),
        )),
        EnemyKind("Iron Warden", 90, 15, (
            (("shield", 15), ("attack", 6, 1)),
            (("attack", 14, 1),),
            (("crack", 2), ("attack", 9, 1)),
        )),
        EnemyKind("Hollow Duelist", 66, 0, (
            (("attack", 8, 2),),
            (("crack", 1), ("attack", 11, 1)),
            (("might", 3), ("shield", 6)),
        )),
    ]
}
ENEMY_NAMES: tuple[str, ...] = tuple(ENEMIES)


def describe_effect(effect: Effect) -> str:
    kind = effect[0]
    if kind == "attack":
        _, damage, hits = effect
        return f"Attack {damage}" + (f" x{hits}" if hits > 1 else "")
    if kind == "shield":
        return f"Gain {effect[1]} Shield"
    if kind == "burn":
        return f"Apply {effect[1]} Burn to you"
    if kind == "crack":
        return f"Apply {effect[1]} Cracked to you"
    if kind == "might":
        return f"Gain {effect[1]} Might"
    raise ValueError(f"unknown effect {effect!r}")


def describe_intent(intent: Intent) -> str:
    return ", then ".join(describe_effect(e) for e in intent)
