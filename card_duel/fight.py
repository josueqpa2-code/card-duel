"""Full multi-turn fights.

The agent sends one action per step ("play <card>" or "end turn") and gets
the new state back as text. The draw pile's order is never shown, only its
contents. Reward at the end:
    win:  0.5 + 0.5 * (HP left / max HP)      -> between 0.5 and 1.0
    loss: 0.4 * (share of enemy HP removed)   -> below 0.4
so any win scores above any loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .enemies import ENEMIES, ENEMY_NAMES, Intent, describe_intent
from .engine import (Battle, Fighter, card_error, end_player_turn, enemy_turn, incoming_damage,
                     play_card, start_player_turn)
from .rng import Rng, mix
from .text import RULES, card_reference, list_cards, parse_action

FIGHT_STREAM = 0x46494748
TEST_EVERY = 5
PLAYER_HP = 50
MAX_TURNS = 30
MAX_STEPS = 400
BASE_DECK = ("Jab", "Jab", "Jab", "Guard", "Guard", "Guard")
EXTRA_POOL = [
    ("Jab", 2), ("Flurry", 3), ("Crack", 3), ("Finisher", 2), ("Cinder Strike", 2),
    ("Shatter", 2), ("Spark", 2), ("Guard", 1), ("Bulwark", 2), ("Ignite", 3),
    ("Stoke", 1), ("Wind Up", 2), ("Scout", 2), ("Overclock", 1), ("Hone", 2),
]
EXTRA_CARDS = 4

ACTION_HELP = ("Reply with exactly one action inside <action></action>: "
               "<action>play CARD NAME</action> or <action>end turn</action>.")


def _cards(n: int) -> str:
    return f"{n} card" if n == 1 else f"{n} cards"


@dataclass(frozen=True)
class FightSpec:
    seed: int
    split: str
    enemy: str
    deck: tuple[str, ...]
    intent_offset: int

    @property
    def id(self) -> str:
        return f"fight-{self.seed}"


def make_fight(seed: int) -> FightSpec:
    rng = Rng(mix(FIGHT_STREAM, seed))
    enemy = rng.choice(ENEMY_NAMES)
    deck = tuple(sorted(BASE_DECK + tuple(rng.weighted(EXTRA_POOL) for _ in range(EXTRA_CARDS))))
    offset = rng.below(len(ENEMIES[enemy].pattern))
    digest = int(hashlib.sha256(f"fight-{seed}".encode()).hexdigest()[:8], 16)
    split = "test" if digest % TEST_EVERY == 0 else "train"
    return FightSpec(seed, split, enemy, deck, offset)


def fight_dataset(split: str = "train", n: int = 100, start_seed: int = 0) -> list[FightSpec]:
    out, seed = [], start_seed
    while len(out) < n:
        spec = make_fight(seed)
        seed += 1
        if spec.split == split:
            out.append(spec)
    return out


class FightEnv:
    def __init__(self, spec: FightSpec | int, max_turns: int = MAX_TURNS, max_steps: int = MAX_STEPS):
        self.spec = spec if isinstance(spec, FightSpec) else make_fight(spec)
        self.max_turns = max_turns
        self.max_steps = max_steps
        self.reset()

    # ------------------------------------------------------------- state

    def reset(self) -> str:
        kind = ENEMIES[self.spec.enemy]
        deck = list(self.spec.deck)
        Rng(mix(FIGHT_STREAM, self.spec.seed, 2)).shuffle(deck)
        self.battle = Battle(
            player=Fighter("You", PLAYER_HP, PLAYER_HP),
            enemy=Fighter(kind.name, kind.max_hp, kind.max_hp, shield=kind.start_shield),
            energy=0, max_energy=3, draw_pile=deck, turn=0,
            seed=mix(FIGHT_STREAM, self.spec.seed, 3),
        )
        self.intent_index = self.spec.intent_offset
        self.steps = 0
        self.illegal = 0
        self.done = False
        self.timed_out = False
        self.history: list[str] = []
        start_player_turn(self.battle)
        return self.intro()

    @property
    def intent(self) -> Intent:
        pattern = ENEMIES[self.spec.enemy].pattern
        return pattern[self.intent_index % len(pattern)]

    @property
    def won(self) -> bool:
        return self.battle.outcome() == "win"

    def reward(self) -> float:
        b = self.battle
        if b.outcome() == "win":
            return 0.5 + 0.5 * max(b.player.hp, 0) / b.player.max_hp
        removed = (b.enemy.max_hp - max(b.enemy.hp, 0)) / b.enemy.max_hp
        return 0.4 * removed

    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        b = self.battle
        return [f"play {n}" for n in sorted(set(b.hand)) if card_error(b, n) is None] + ["end turn"]

    # ------------------------------------------------------------- text

    def intro(self) -> str:
        return f"""You are playing Card Duel, a turn-based card game. Win the fight while keeping as much HP as you can.

{RULES}

FIGHT
- You have {PLAYER_HP} HP and 3 energy per turn. The fight is lost if it lasts more than {self.max_turns} turns.
- The enemy repeats a fixed cycle of intents. Its next intent is always shown.
- Your score: a win gives 0.5 plus up to 0.5 more for HP kept; a loss scores below 0.4.

YOUR DECK
{card_reference(self.spec.deck)}

{ACTION_HELP}

{self.state_text()}"""

    def state_text(self) -> str:
        b = self.battle
        p, e = b.player, b.enemy
        incoming = incoming_damage(b, self.intent)
        hit_note = f" ({incoming} damage to you before Shield)" if incoming else ""
        return f"""TURN {b.turn}
You: HP {p.hp}/{p.max_hp}, Shield {p.shield}, Energy {b.energy}, Might {p.might}, Charge {b.charge}, Burn {p.burn}, Cracked {p.cracked}
Enemy ({e.name}): HP {e.hp}/{e.max_hp}, Shield {e.shield}, Might {e.might}, Burn {e.burn}, Cracked {e.cracked}
Enemy intent: {describe_intent(self.intent)}{hit_note}
Hand: {list_cards(b.hand)}
Draw pile ({_cards(len(b.draw_pile))}, order hidden): {list_cards(b.draw_pile)}
Discard pile ({_cards(len(b.discard))}): {list_cards(b.discard)}
Exhausted: {list_cards(b.exhausted)}"""

    def _final_text(self) -> str:
        b = self.battle
        if b.outcome() == "win":
            head = f"YOU WON on turn {b.turn} with {b.player.hp}/{b.player.max_hp} HP."
        elif self.timed_out:
            head = "YOU LOST: the fight went past the turn limit."
        elif self.steps >= self.max_steps and b.outcome() is None:
            head = "YOU LOST: too many actions."
        else:
            head = f"YOU LOST on turn {b.turn}. The enemy had {max(b.enemy.hp, 0)} HP left."
        return f"{head} Score: {self.reward():.3f}"

    # ------------------------------------------------------------- acting

    def step(self, text: str) -> tuple[str, float, bool, dict]:
        """Apply one action written as text. Returns (observation, reward, done, info)."""
        if self.done:
            return self._final_text(), self.reward(), True, {"error": "the fight is over"}
        self.steps += 1
        action, err = parse_action(text)
        if action is not None and action[0] == "play":
            err = card_error(self.battle, action[1])
        if err:
            self.illegal += 1
            self.history.append(f"illegal: {text.strip()[:60]}")
            if self.steps >= self.max_steps:
                return self._finish({"error": err})
            obs = f"Illegal action: {err}. Nothing changed.\n{ACTION_HELP}\n\n{self.state_text()}"
            return obs, 0.0, False, {"error": err}

        b = self.battle
        if action[0] == "play":
            name = action[1]
            enemy_hp = b.enemy.hp
            play_card(b, name)
            self.history.append(f"T{b.turn} play {name}")
            note = f"You played {name}."
            if b.enemy.hp < enemy_hp:
                note += f" The enemy lost {enemy_hp - b.enemy.hp} HP."
        else:
            note = self._end_turn()
        if b.outcome() is not None or self.timed_out or self.steps >= self.max_steps:
            return self._finish({}, note)
        return f"{note}\n\n{self.state_text()}", 0.0, False, {}

    def _end_turn(self) -> str:
        b = self.battle
        self.history.append(f"T{b.turn} end turn")
        intent = self.intent
        hp, shield, enemy_hp = b.player.hp, b.player.shield, b.enemy.hp
        end_player_turn(b)
        enemy_turn(b, intent)
        self.intent_index += 1
        parts = []
        if b.enemy.hp < enemy_hp:
            parts.append(f"The enemy lost {enemy_hp - b.enemy.hp} HP to Burn.")
        if b.enemy.hp > 0:
            parts.append(f"The enemy did: {describe_intent(intent)}.")
            lost = hp - b.player.hp
            if lost or shield:
                parts.append(f"You lost {lost} HP" + (f" (you had {shield} Shield)." if shield else "."))
        if b.outcome() is None:
            hp_before_upkeep = b.player.hp
            start_player_turn(b)
            if b.player.hp < hp_before_upkeep:
                parts.append(f"You lost {hp_before_upkeep - b.player.hp} HP to Burn.")
            if b.turn > self.max_turns and b.outcome() is None:
                self.timed_out = True
        return " ".join(parts) or "Your turn ended."

    def _finish(self, info: dict, note: str = "") -> tuple[str, float, bool, dict]:
        self.done = True
        info = {**info, "won": self.won, "turns": self.battle.turn, "hp_left": max(self.battle.player.hp, 0),
                "illegal_actions": self.illegal}
        text = f"{note}\n\n{self._final_text()}" if note else self._final_text()
        return text, self.reward(), True, info
