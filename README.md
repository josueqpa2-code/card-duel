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
| lookahead (2 cards) | 89% | 68% | 42% | 69% |
| lookahead (3 cards) | 99% | 88% | 73% | 88% |
| exact solver | 100% | 100% | 100% | 100% |

Fights: 200 from the test split.

| Player | Win rate | Mean reward | Mean HP left (wins) |
|---|---|---|---|
| random | 0% | 0.146 | 0.0 |
| greedy | 43% | 0.443 | 9.3 |
| planner | 78% | 0.554 | 11.4 |
| planner (2 turns) | 82% | 0.560 | 10.9 |

- **Puzzle difficulty is measured, not guessed.** "Easy" means a greedy player solves it.
  "Medium" and "hard" puzzles defeat the greedy player, and in hard ones fewer than 2% of all
  possible lines win.
- **Hard puzzles need deep planning.** A player that looks 3 cards ahead before each play still
  fails 27% of them. Only full search solves them all.
- **The fight planners are not the ceiling.** "planner" searches every line of the current turn
  and scores it after the enemy's reply. "planner (2 turns)" re-scores its best 6 lines by also
  planning the next turn over 6 sampled draws.
- **Extra search helps fights only a little:** 78% to 82% wins. Mean HP kept stays near 11 of
  50, so a strong player can gain mostly by finishing fights with more HP.
- **No reference player peeks.** None of them can see the hidden draw order, and a test
  checks that reordering the draw pile never changes a choice.
- No language-model results are included yet.

## Quick start

```bash
python -m card_duel puzzle 12            # print a puzzle prompt
python -m card_duel solve 12             # its answer, and how rare winning lines are
python -m card_duel check 12 "Wind Up, Jab"
python -m card_duel fight 3              # play a fight yourself
python -m card_duel benchmark 300 200
python -m unittest discover -s tests -t .   # 69 tests
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

## Score a model

`eval-model` works with any OpenAI-compatible chat API, using only the standard library. It
reads `OPENAI_API_KEY`, plus `OPENAI_BASE_URL` for other providers.

```bash
python -m card_duel eval-model MODEL --task puzzle --n 100 --difficulty hard
python -m card_duel eval-model MODEL --task fight --n 20 --workers 2
```

- **Output:** a summary is printed (solved by difficulty, legal-answer rate, or fight win
  rate and mean reward). Every task is appended to `outputs/<model>-<task>-<split>.jsonl`
  with the model's replies.
- **Fights:** the model sees the opening message plus the last 16 messages (`--window`).
  Each observation repeats the full state.
- **Failures:** rate limits and server errors are retried with backoff. A request that still
  fails is recorded as an API error, not as a wrong answer.
- **Testing:** the evaluator is tested against a fake local server (`tests/test_llm_eval.py`).
  No real model has been scored yet.

To get the raw datasets as JSONL (prompt, reference answer and full puzzle state):

```bash
python -m card_duel export data 2000 300
```

Fine-tuning data comes from the train split only, as chat-format JSONL:

```bash
python -m card_duel export-sft data 1000 200
```

- **Puzzles:** each example ends with a worked solution. Every step comes from the engine, so
  the reasoning is correct by construction, followed by the `<answer>` tag.
- **Fights:** winning games played by the two-turn planner. 19 of the first 20 train fights
  are wins.
- **Checks:** tests confirm that every worked solution scores 1.0 and that fight transcripts
  replay exactly.

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

**verifiers v1 (current).** `environments/` holds two Hub packages, each built in the same shape
as an example in the verifiers repository:

| Package | Exports | Shape | Modelled on |
|---|---|---|---|
| `environments/card_duel_puzzles` | `CardDuelPuzzleTaskset` | single turn; `@vf.reward` reads `trace.last_reply` | the `gsm8k` example |
| `environments/card_duel_fights` | `CardDuelFightTaskset`, `CardDuelFightEnv` | multi-turn; the game engine plays the user | the `textarena` (Wordle) taskset |

```bash
uv pip install -e environments/card_duel_puzzles
uv run vf-validate card-duel-puzzles -n 1
uv run vf-run card-duel-puzzles -n 5 -r 1
```

- **Config:** `split` (train/test) and `num_tasks` for both packages. The puzzles package also
  takes `difficulty` and `partial_credit`.
- **Before pushing to the Hub:** each `pyproject.toml` points `card-duel` at this repository
  through `[tool.uv.sources]`. Change that to a published release or a git URL first.

**verifiers v0 (legacy).** The Hub now lists v0 environments as Legacy.

```python
import card_duel   # pip install "card-duel[verifiers]"
env = card_duel.load_environment(task="puzzle", num_train=2000, num_eval=200, difficulty=None)
env = card_duel.load_environment(task="fight", num_train=2000, num_eval=200, max_turns=150)
```

**Status: neither adapter has been run against a real verifiers install yet.**

- **What the tests cover:** both are tested against stand-in modules
  (`tests/test_hub_v1.py`, `tests/test_verifiers_adapter.py`). Those tests check loading,
  scoring, and that every observation the multi-turn loop sends matches the engine.
- **What they don't cover:** compatibility with a real verifiers release.
- **Saving results:** the v0 adapter keeps only plain JSON in rollout state. It rebuilds the
  fight from its seed and the actions taken so far.

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
