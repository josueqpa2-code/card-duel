"""Game rules. This module is the single source of truth for Card Duel.

Damage from one hit:  (base + attacker Might + Charge), x1.5 rounded down if
the target is Cracked, then the target's Shield absorbs what it can.
Start of a fighter's own turn: its Shield is removed, then its Burn ticks
(lose Burn HP, then Burn goes down by 1).
End of a fighter's own turn: its Cracked goes down by 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .cards import ATTACK, CARDS
from .enemies import Effect, Intent
from .rng import Rng, mix

HAND_LIMIT = 10
DRAW_PER_TURN = 5


class IllegalAction(Exception):
    pass


@dataclass
class Fighter:
    name: str
    hp: int
    max_hp: int
    shield: int = 0
    burn: int = 0
    cracked: int = 0
    might: int = 0

    def copy(self) -> "Fighter":
        return replace(self)

    def key(self) -> tuple:
        return (self.hp, self.shield, self.burn, self.cracked, self.might)


@dataclass
class Battle:
    player: Fighter
    enemy: Fighter
    energy: int
    max_energy: int = 3
    hand: list[str] = field(default_factory=list)
    draw_pile: list[str] = field(default_factory=list)  # index 0 is the top card
    discard: list[str] = field(default_factory=list)
    exhausted: list[str] = field(default_factory=list)
    charge: int = 0
    attacks_this_turn: int = 0
    turn: int = 1
    seed: int = 0
    shuffles: int = 0
    # A planning copy: the planner may not see the draw order, so draws are
    # only counted in pending_draws instead of taking real cards.
    blind: bool = False
    pending_draws: int = 0

    def clone(self) -> "Battle":
        return Battle(
            player=self.player.copy(),
            enemy=self.enemy.copy(),
            energy=self.energy,
            max_energy=self.max_energy,
            hand=list(self.hand),
            draw_pile=list(self.draw_pile),
            discard=list(self.discard),
            exhausted=list(self.exhausted),
            charge=self.charge,
            attacks_this_turn=self.attacks_this_turn,
            turn=self.turn,
            seed=self.seed,
            shuffles=self.shuffles,
            blind=self.blind,
            pending_draws=self.pending_draws,
        )

    def blind_copy(self) -> "Battle":
        """A copy for planning that hides the draw and discard order."""
        b = self.clone()
        b.blind = True
        b.draw_pile = []
        b.discard = []
        return b

    def key(self) -> tuple:
        return (
            self.player.key(),
            self.enemy.key(),
            self.energy,
            tuple(sorted(self.hand)),
            tuple(self.draw_pile),
            tuple(sorted(self.discard)),
            self.charge,
            self.attacks_this_turn,
            self.shuffles,
            self.pending_draws,
        )

    def outcome(self) -> str | None:
        if self.player.hp <= 0:
            return "loss"
        if self.enemy.hp <= 0:
            return "win"
        return None


# ----------------------------------------------------------------- damage

def take_damage(target: Fighter, amount: int) -> int:
    """Apply damage through Shield. Returns HP actually lost."""
    amount = max(0, amount)
    absorbed = min(target.shield, amount)
    target.shield -= absorbed
    lost = amount - absorbed
    target.hp -= lost
    return lost


def hit_damage(base: int, attacker_might: int, charge: int, target_cracked: int) -> int:
    dmg = base + attacker_might + charge
    if target_cracked > 0:
        dmg = dmg * 3 // 2
    return max(0, dmg)


def _player_hits(b: Battle, base: int, hits: int = 1) -> None:
    for _ in range(hits):
        if b.enemy.hp <= 0:
            return
        take_damage(b.enemy, hit_damage(base, b.player.might, b.charge, b.enemy.cracked))


# ----------------------------------------------------------------- cards

def _jab(b: Battle) -> None:
    _player_hits(b, 6)


def _flurry(b: Battle) -> None:
    _player_hits(b, 2, 3)


def _crack(b: Battle) -> None:
    _player_hits(b, 7)
    b.enemy.cracked += 2


def _finisher(b: Battle) -> None:
    _player_hits(b, 4 + 3 * b.attacks_this_turn)


def _cinder_strike(b: Battle) -> None:
    _player_hits(b, 4 + b.enemy.burn)


def _shatter(b: Battle) -> None:
    b.enemy.shield = 0
    _player_hits(b, 3)


def _spark(b: Battle) -> None:
    _player_hits(b, 3)


def _guard(b: Battle) -> None:
    b.player.shield += 5


def _bulwark(b: Battle) -> None:
    b.player.shield += 12


def _ignite(b: Battle) -> None:
    b.enemy.burn += 3


def _stoke(b: Battle) -> None:
    b.enemy.burn *= 2


def _wind_up(b: Battle) -> None:
    b.charge += 4


def _scout(b: Battle) -> None:
    draw(b, 2)


def _overclock(b: Battle) -> None:
    b.energy += 2
    b.player.hp -= 3


def _hone(b: Battle) -> None:
    b.player.might += 1


_EFFECTS = {
    "Jab": _jab,
    "Flurry": _flurry,
    "Crack": _crack,
    "Finisher": _finisher,
    "Cinder Strike": _cinder_strike,
    "Shatter": _shatter,
    "Spark": _spark,
    "Guard": _guard,
    "Bulwark": _bulwark,
    "Ignite": _ignite,
    "Stoke": _stoke,
    "Wind Up": _wind_up,
    "Scout": _scout,
    "Overclock": _overclock,
    "Hone": _hone,
}
assert set(_EFFECTS) == set(CARDS), "every card needs an effect"


def card_error(b: Battle, name: str) -> str | None:
    """Why `name` can't be played now, or None if it can."""
    if b.outcome() is not None:
        return "the fight is already over"
    if name not in CARDS:
        return f"there is no card called {name!r}"
    if name not in b.hand:
        return f"{name} is not in your hand"
    cost = CARDS[name].cost
    if cost > b.energy:
        return f"{name} costs {cost} energy but you have {b.energy}"
    return None


def playable_cards(b: Battle) -> list[str]:
    return [n for n in sorted(set(b.hand)) if card_error(b, n) is None]


def play_card(b: Battle, name: str) -> None:
    err = card_error(b, name)
    if err:
        raise IllegalAction(err)
    card = CARDS[name]
    b.energy -= card.cost
    b.hand.remove(name)
    _EFFECTS[name](b)
    if card.kind == ATTACK:
        b.charge = 0
        b.attacks_this_turn += 1
    (b.exhausted if card.exhaust else b.discard).append(name)


def draw(b: Battle, n: int) -> None:
    for _ in range(n):
        if len(b.hand) + b.pending_draws >= HAND_LIMIT:
            return
        if b.blind:
            b.pending_draws += 1
            continue
        if not b.draw_pile:
            if not b.discard:
                return
            b.draw_pile = b.discard
            b.discard = []
            Rng(mix(b.seed, b.shuffles)).shuffle(b.draw_pile)
            b.shuffles += 1
        b.hand.append(b.draw_pile.pop(0))


# ----------------------------------------------------------------- turns

def start_of_turn(f: Fighter) -> int:
    """Upkeep at the start of a fighter's own turn. Returns HP lost to Burn."""
    f.shield = 0
    if f.burn > 0:
        lost = f.burn
        f.hp -= lost
        f.burn -= 1
        return lost
    return 0


def end_of_turn(f: Fighter) -> None:
    if f.cracked > 0:
        f.cracked -= 1


def end_player_turn(b: Battle) -> None:
    b.discard.extend(b.hand)
    b.hand = []
    b.charge = 0
    b.attacks_this_turn = 0
    end_of_turn(b.player)


def start_player_turn(b: Battle) -> None:
    b.turn += 1
    start_of_turn(b.player)
    if b.player.hp <= 0:
        return
    b.energy = b.max_energy
    draw(b, DRAW_PER_TURN)


def apply_enemy_effect(b: Battle, effect: Effect) -> None:
    kind = effect[0]
    if kind == "attack":
        _, damage, hits = effect
        for _ in range(hits):
            if b.player.hp <= 0:
                return
            take_damage(b.player, hit_damage(damage, b.enemy.might, 0, b.player.cracked))
    elif kind == "shield":
        b.enemy.shield += effect[1]
    elif kind == "burn":
        b.player.burn += effect[1]
    elif kind == "crack":
        b.player.cracked += effect[1]
    elif kind == "might":
        b.enemy.might += effect[1]
    else:
        raise ValueError(f"unknown effect {effect!r}")


def enemy_turn(b: Battle, intent: Intent) -> None:
    """The whole enemy turn: upkeep, its intent, then end-of-turn upkeep."""
    start_of_turn(b.enemy)
    if b.enemy.hp <= 0:
        return
    for effect in intent:
        apply_enemy_effect(b, effect)
        if b.player.hp <= 0:
            return
    end_of_turn(b.enemy)


def incoming_damage(b: Battle, intent: Intent) -> int:
    """Damage the intent would deal to the player before Shield."""
    cracked = b.player.cracked
    might = b.enemy.might
    total = 0
    for effect in intent:
        if effect[0] == "attack":
            total += hit_damage(effect[1], might, 0, cracked) * effect[2]
        elif effect[0] == "crack":
            cracked += effect[1]
        elif effect[0] == "might":
            might += effect[1]
    return total
