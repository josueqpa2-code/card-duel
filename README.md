# Card Duel

An original turn-based card game built as a training and evaluation environment for AI agents.
Every reward comes from the game engine itself, so it can be checked exactly, and every task
is fully determined by a seed. It is pure Python with no dependencies.

It has two task types:

- **Puzzles.** Single-turn "win this turn" problems. An exhaustive solver sets the enemy's HP to
  exactly the most damage any line of play can deal, so a solution always exists and only the
  best play wins. The reward is checked exactly.
- **Fights.** Full multi-turn battles against an enemy whose moves follow a fixed, visible cycle.
  The agent plays one card per message, and the score is whether it won plus how much HP it kept.

## Baselines

These come from `python -m card_duel benchmark 300 200` on the held-out test split.

Puzzles: 300 from the test split (91 easy, 145 medium, 64 hard). Share solved:

| Player | easy | medium | hard | all |
|---|---|---|---|---|
| random | 10% | 6% | 0% | 6% |
| greedy | 100% | 0% | 0% | 30% |
| exact solver | 100% | 100% | 100% | 100% |

Fights: 200 from the test split.

| Player | Win rate | Mean reward | Mean HP left (wins) |
|---|---|---|---|
| random | 0% | 0.146 | 0.0 |
| greedy | 43% | 0.443 | 9.3 |
| planner | 78% | 0.554 | 11.4 |

- **Puzzle difficulty is measured, not guessed.** "Easy" means a greedy player solves it.
  "Medium" and "hard" puzzles defeat the greedy player, and in hard ones fewer than 2% of all
  possible lines win.
- **The fight planner is not the ceiling.** It searches every line of the current turn but looks
  only one enemy move ahead, so there is headroom above it.
- No language-model results are included yet.

## Quick start

```bash
python -m card_duel puzzle 12            # print a puzzle prompt
python -m card_duel solve 12             # its answer, and how rare winning lines are
python -m card_duel check 12 "Wind Up, Jab"
python -m card_duel fight 3              # play a fight yourself
python -m card_duel benchmark 300 200
python -m unittest discover -s tests -t .
```

```python
from card_duel import dataset, score_answer, FightEnv

p = dataset("test", 1, difficulty="hard")[0]
prompt = p.prompt()                                  # text for the model
result = score_answer(p, "<answer>Wind Up, Jab</answer>")
result.reward, result.solved, result.legal, result.error

env = FightEnv(seed=7)
obs = env.intro()
obs, reward, done, info = env.step("<action>play Jab</action>")
```

## Tasks in detail

**Puzzle answers.** The model writes a play order inside `<answer></answer>`, for example
`<answer>Wind Up, Flurry, Jab</answer>`. The parser accepts case and spacing variations,
numbering, arrows and a leading "play". If the answer has several tags, the last one counts.

- **Reward:** 1.0 if the enemy is defeated. Anything else scores 0.0: an illegal play, an
  unknown card, defeating yourself, or running out of damage.
- **Partial credit (optional):** with `partial_credit=True`, a legal line that falls short
  scores 0.5 × the share of the enemy's HP removed.
- **Extra cards:** any cards listed after the winning play are ignored.

**Fight actions.** Each turn the model sends `<action>play CARD</action>` or
`<action>end turn</action>`.

- **Bad actions:** an illegal or unreadable action changes nothing. The model gets an error and
  the unchanged state.
- **Reward at the end:**
  - a win scores 0.5 + 0.5 × (HP left ÷ max HP);
  - a loss scores 0.4 × (share of enemy HP removed);
  - so any win beats any loss.
- **Limits:** a fight lasting more than 30 turns or 400 actions is lost.

**Splits.** About 1 in 5 puzzles is assigned to the test split by a hash of the puzzle's
content, so an identical puzzle can never appear in both splits. Fights are split by a hash of
the seed.

## Use with verifiers / Prime Intellect Environments Hub

```bash
pip install "card-duel[verifiers]"
```

```python
import card_duel
env = card_duel.load_environment(task="puzzle", num_train=2000, num_eval=200, difficulty=None)
env = card_duel.load_environment(task="fight", num_train=2000, num_eval=200, max_turns=150)
```

- **Status: not yet run against a real verifiers install.** The adapter in
  `card_duel/verifiers_env.py` was written for verifiers 0.1.x. Its reward functions and the
  multi-turn loop are tested with stand-in modules (`tests/test_verifiers_adapter.py`).
- **Saving results works.** Rollout state holds only plain JSON: the fight is rebuilt from its
  seed and the actions taken so far.

## Why agents find it hard

The rules are short, but small details decide the puzzles. Each of these is covered by a test:

- **Charge** goes into the *next attack card* only. Playing Wind Up before Jab rather than before
  Flurry changes the damage from 10 to 18.
- **Cracked** from Crack applies *after* Crack's own hit.
- **Burn** ticks at the start of the enemy's turn, before it acts, so Burn damage counts toward a
  win this turn. Stoke doubles it and Cinder Strike scales with it.
- **Shield** soaks up attack damage but not Burn. Removing it with Shatter, or going around it
  with Burn, changes how much damage a line deals.
- **Finisher** counts only attacks played *earlier* in the turn.
- **Overclock** costs 3 HP, which is fatal at 3 HP or less.
- **Scout** draws from a draw pile whose order is shown in puzzles. Fights never show the draw
  order, and the reference players never peek at it either.
- **Decoy cards** such as Guard and Bulwark, and rounding down on every Cracked hit, punish
  answers that are only roughly right.

`examples/` has sample puzzles at each difficulty, a walkthrough of a hard puzzle that the greedy
player fails, and full fight transcripts from the planner and greedy players. Regenerate them
with `python -m card_duel examples examples`.

## Design notes

- **One rules engine.** `card_duel/engine.py` is the only implementation of the rules. The
  prompts, solver, baselines and scorer all use it.
- **Same tasks everywhere.** Python's `random` module can produce different sequences in
  different versions, so the package uses its own seeded generator (SplitMix64). A seed gives the
  same task on any machine.
- **Exact solver.** The solver in `card_duel/solver.py` searches every order of play, merging
  identical cards. It returns the best value, the shortest best line, and the number of winning
  and total lines.
- **Original game.** Card Duel's rules, cards and enemies are its own. It uses no other game's
  code, data or assets.

## Licence

MIT
