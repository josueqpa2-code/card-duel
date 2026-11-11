"""Everything an agent reads or writes: rules text, state rendering, parsing."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .cards import CARDS, lookup

RULES = """\
RULES
- You fight one enemy. Each card costs the energy shown in [brackets]. You can't play a card you can't pay for.
- Attack damage per hit = base damage + your Might + your Charge. If the target is Cracked, multiply by 1.5 and round down. The target's Shield absorbs damage before its HP.
- A card that hits several times works out each hit separately, with the same bonuses.
- Charge is added to every hit of the next attack card you play, then it is used up. Unused Charge is lost when your turn ends.
- Might is added to every hit of that fighter's attacks for the rest of the fight. Enemies can have Might too.
- Cracked N: the fighter takes 1.5x attack damage (rounded down). It goes down by 1 at the end of that fighter's own turn.
- Burn N: at the start of that fighter's own turn it loses N HP (Shield does not stop this), then its Burn goes down by 1.
- Shield absorbs damage. A fighter loses all of its Shield at the start of its own turn.
- A played card goes to the discard pile, unless it says Exhaust: then it is removed for the rest of the fight.
- You draw from the top of the draw pile. If the draw pile is empty, the discard pile is shuffled to form a new one. Your hand holds at most 10 cards; extra draws are skipped.
- Your turn: you lose your Shield, your Burn ticks, energy refills to its maximum and you draw 5 cards. Then you play cards. When you end your turn, your hand is discarded and your Cracked ticks.
- Enemy turn: it loses its Shield, its Burn ticks, it does its intent (shown to you in advance), then its Cracked ticks.
- A fighter whose HP reaches 0 is defeated at once."""


def card_reference(names: Iterable[str]) -> str:
    unique = sorted(set(names), key=lambda n: list(CARDS).index(n))
    return "\n".join(f"- {CARDS[n].describe()}" for n in unique)


def list_cards(names: Iterable[str], keep_order: bool = False) -> str:
    names = list(names)
    if not names:
        return "empty"
    if keep_order:
        return ", ".join(names)
    counts = Counter(names)
    return ", ".join(n if counts[n] == 1 else f"{n} x{counts[n]}" for n in sorted(counts))


# ----------------------------------------------------------------- parsing

def extract_tag(text: str, tag: str) -> str | None:
    """Content of the last <tag>...</tag> in text, or None."""
    found = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
    return found[-1] if found else None


_SPLIT = re.compile(r"[,\n;]|->|=>|→|\bthen\b", flags=re.I)
_PREFIX = re.compile(r"^\s*(?:\d+\s*[.):]\s*|[-*•]\s*)")
_VERB = re.compile(r"^(?:play|use|cast)\s+", flags=re.I)
_END_WORDS = {"end", "end turn", "endturn", "pass", "done"}


def parse_line(answer: str) -> tuple[list[str], str | None]:
    """Read a play order like "Wind Up, Jab, Jab". Returns (cards, error)."""
    if answer.strip().lower().strip(".") in _END_WORDS | {"", "none", "nothing"}:
        return [], None
    names: list[str] = []
    for part in _SPLIT.split(answer):
        p = _VERB.sub("", _PREFIX.sub("", part.strip())).strip().strip(".").strip()
        if not p:
            continue
        if p.lower() in _END_WORDS:
            break  # the turn ends here; anything after is not played
        name = lookup(p)
        if name is None:
            return names, f"unknown card {p!r}"
        names.append(name)
    return names, None


def parse_action(text: str) -> tuple[tuple[str, str | None] | None, str | None]:
    """Read one fight action. Returns (("play", name) or ("end", None), error)."""
    raw = extract_tag(text, "action")
    s = (raw if raw is not None else text).strip().strip(".").strip()
    if s.lower() in _END_WORDS:
        return ("end", None), None
    name = lookup(_VERB.sub("", s))
    if name is not None:
        return ("play", name), None
    shown = s if len(s) <= 60 else s[:57] + "..."
    return None, f"could not read an action from {shown!r}"
