# A hard two-turn puzzle (plan-0)

```
You are playing Card Duel, a turn-based card game.

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
- A fighter whose HP reaches 0 is defeated at once.

PUZZLE: WIN WITHIN TWO TURNS
Plan your next two turns so the enemy is defeated before its second action.
- Turn 1: play cards from your hand. Then your turn ends (your hand is discarded) and the enemy takes ONE action: Gain 3 Might, then Gain 6 Shield (0 damage to you before Shield, as things stand now).
- Turn 2: you lose your Shield, your Burn ticks, energy refills to 3 and you draw 5 cards from the top of the draw pile. Play cards, then your turn ends.
- You win if the enemy's HP reaches 0 at any point up to the start of its next turn (when its Burn ticks again). You lose if your HP reaches 0.
- The draw pile's order is shown, and it is long enough that it will not run out.

STATE
You: HP 15/40, Energy 3, Might 0, Charge 0
Enemy (Hollow Duelist): HP 51/66, Shield 4, Burn 2, Cracked 1, Might 0
Hand: Finisher, Flurry, Ignite, Scout, Spark
Draw pile, top card first: Flurry, Jab, Bulwark, Wind Up, Scout, Jab, Stoke, Guard, Wind Up
Discard pile: empty

CARDS
- Jab [1 energy, attack]: Deal 6 damage.
- Flurry [1 energy, attack]: Deal 2 damage 3 times.
- Finisher [2 energy, attack]: Deal 4 damage, plus 3 for each other attack card you played earlier this turn.
- Spark [0 energy, attack]: Deal 3 damage.
- Guard [1 energy, skill]: Gain 5 Shield.
- Bulwark [2 energy, skill]: Gain 12 Shield.
- Ignite [1 energy, skill]: Apply 3 Burn to the enemy.
- Stoke [1 energy, skill]: Double the enemy's Burn. Exhaust.
- Wind Up [0 energy, skill]: Gain 4 Charge.
- Scout [0 energy, skill]: Draw 2 cards. Exhaust.

Answer with the cards for turn 1, then a "|", then the cards for turn 2, all inside <answer></answer>.
Example: <answer>Guard, Jab | Wind Up, Flurry, Jab</answer>
```

**Exact plan:** turn 1: Flurry, Spark, Finisher | turn 2: Jab, Scout, Jab, Wind Up, Flurry

**Most damage now, then the best turn 2:** turn 1: Flurry, Scout, Flurry, Jab, Spark | turn 2: Scout, Stoke, Wind Up, Wind Up, Jab -> 39/51 damage, not enough.
