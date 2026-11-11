# A hard puzzle (puzzle-6)

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

PUZZLE: WIN THIS TURN
Choose which cards to play this turn, and in what order, so the enemy is defeated before it can act.
- When you stop playing cards your turn ends. The enemy's turn then starts: it loses its Shield and its Burn ticks. If that brings its HP to 0, you win.
- There are no later turns. If the enemy is still alive after its Burn ticks, you lose.
- If your own HP reaches 0, you lose.

STATE
You: HP 9/40, Energy 4, Might 0, Charge 0
Enemy (Ash Wisp): HP 30/64, Shield 4, Burn 0, Cracked 0
Hand: Finisher x2, Flurry, Jab, Scout, Spark, Stoke
Draw pile, top card first: Ignite, Spark, Overclock
Discard pile: empty

CARDS
- Jab [1 energy, attack]: Deal 6 damage.
- Flurry [1 energy, attack]: Deal 2 damage 3 times.
- Finisher [2 energy, attack]: Deal 4 damage, plus 3 for each other attack card you played earlier this turn.
- Spark [0 energy, attack]: Deal 3 damage.
- Ignite [1 energy, skill]: Apply 3 Burn to the enemy.
- Stoke [1 energy, skill]: Double the enemy's Burn. Exhaust.
- Scout [0 energy, skill]: Draw 2 cards. Exhaust.
- Overclock [0 energy, skill]: Gain 2 energy. Lose 3 HP. Exhaust.

Answer with the cards to play, in order, separated by commas, inside <answer></answer>.
Example: <answer>Wind Up, Jab, Jab</answer>
```

**Exact answer:** Flurry, Jab, Scout, Spark, Spark, Finisher  
Only 40 of 5217 possible lines win.

**Greedy player:** Flurry, Jab, Spark, Finisher -> 24/30 damage, not enough.
