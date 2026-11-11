"""Card list. The rules text here is shown to agents; engine.py implements it."""

from __future__ import annotations

from dataclasses import dataclass

ATTACK = "attack"
SKILL = "skill"


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    kind: str
    text: str
    exhaust: bool = False

    def describe(self) -> str:
        tail = " Exhaust." if self.exhaust else ""
        return f"{self.name} [{self.cost} energy, {self.kind}]: {self.text}{tail}"


_ALL = [
    Card("Jab", 1, ATTACK, "Deal 6 damage."),
    Card("Flurry", 1, ATTACK, "Deal 2 damage 3 times."),
    Card("Crack", 2, ATTACK, "Deal 7 damage, then apply 2 Cracked."),
    Card("Finisher", 2, ATTACK,
         "Deal 4 damage, plus 3 for each other attack card you played earlier this turn."),
    Card("Cinder Strike", 1, ATTACK, "Deal 4 damage, plus 1 for each Burn on the enemy."),
    Card("Shatter", 1, ATTACK, "Remove all of the enemy's Shield, then deal 3 damage."),
    Card("Spark", 0, ATTACK, "Deal 3 damage."),
    Card("Guard", 1, SKILL, "Gain 5 Shield."),
    Card("Bulwark", 2, SKILL, "Gain 12 Shield."),
    Card("Ignite", 1, SKILL, "Apply 3 Burn to the enemy."),
    Card("Stoke", 1, SKILL, "Double the enemy's Burn.", exhaust=True),
    Card("Wind Up", 0, SKILL, "Gain 4 Charge."),
    Card("Scout", 0, SKILL, "Draw 2 cards.", exhaust=True),
    Card("Overclock", 0, SKILL, "Gain 2 energy. Lose 3 HP.", exhaust=True),
    Card("Hone", 1, SKILL, "Gain 1 Might.", exhaust=True),
]

CARDS: dict[str, Card] = {c.name: c for c in _ALL}
CARD_NAMES: tuple[str, ...] = tuple(CARDS)


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


_BY_NORM = {_norm(name): name for name in CARDS}


def lookup(text: str) -> str | None:
    """Match a card name loosely: case, spaces and punctuation are ignored."""
    return _BY_NORM.get(_norm(text))
