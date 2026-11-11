"""Command line: python -m card_duel <command>

  puzzle SEED              print the prompt for a puzzle
  solve SEED               print the puzzle's answer and how rare winning lines are
  check SEED "A, B, C"     score a play order for a puzzle
  fight SEED [LEVEL]       play a fight yourself in the terminal (level 0-3)
  benchmark [P] [F]        baseline scores on P test puzzles and F test fights
  examples DIR             write sample tasks and a full fight transcript
  export DIR [TRAIN] [TEST]  write the puzzle and fight datasets as JSONL
  export-sft DIR [P] [F]   write fine-tuning chats: worked puzzle solutions, winning fights
  eval-model MODEL [...]   score an OpenAI-compatible chat model (--help for options)
  rules [FILE]             print or write the rules, cards and enemies as Markdown
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from .baselines import GreedyPlayer, PlannerPlayer, RandomPlayer, TwoTurnPlanner, run_fight
from .fight import MAX_LEVEL, FightEnv, fight_dataset, make_fight
from .puzzles import dataset, generate, greedy_line, lookahead_line, random_line, score_answer, score_line


def _puzzle(args):
    print(generate(int(args[0])).prompt())


def _solve(args):
    p = generate(int(args[0]))
    print(f"{p.id} ({p.split}, {p.difficulty}): {', '.join(p.solution)}")
    print(f"{p.winning_lines} of {p.total_lines} possible lines win "
          f"({100 * p.winning_lines / p.total_lines:.1f}%).")


def _check(args):
    p = generate(int(args[0]))
    r = score_answer(p, f"<answer>{args[1]}</answer>")
    print(f"solved={r.solved} legal={r.legal} damage={r.damage}/{p.enemy_hp} reward={r.reward}"
          + (f" error={r.error}" if r.error else ""))


def _fight(args):
    env = FightEnv(make_fight(int(args[0]), int(args[1]) if len(args) > 1 else 0))
    print(env.intro())
    while not env.done:
        try:
            line = input("\nYour action (e.g. 'play Jab' or 'end turn'): ")
        except EOFError:
            return
        obs, *_ = env.step(line)
        print("\n" + obs)


def benchmark(n_puzzles: int = 300, n_fights: int = 200) -> str:
    puzzles = dataset("test", n_puzzles)
    players = {
        "random": lambda p: random_line(p, 0),
        "greedy": greedy_line,
        "lookahead (2 cards)": lambda p: lookahead_line(p, 2),
        "lookahead (3 cards)": lambda p: lookahead_line(p, 3),
        "exact solver": lambda p: list(p.solution),
    }
    by_diff = defaultdict(list)
    for p in puzzles:
        by_diff[p.difficulty].append(p)
    order = [d for d in ("easy", "medium", "hard") if d in by_diff]
    lines = [f"Puzzles: {len(puzzles)} from the test split "
             f"({', '.join(f'{len(by_diff[d])} {d}' for d in order)}). Share solved:", "",
             "| Player | " + " | ".join(order) + " | all |", "|---" * (len(order) + 2) + "|"]
    for name, fn in players.items():
        cells = []
        for d in order:
            group = by_diff[d]
            cells.append(f"{sum(score_line(p, fn(p)).solved for p in group) / len(group):.0%}")
        total = sum(score_line(p, fn(p)).solved for p in puzzles) / len(puzzles)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {total:.0%} |")
    specs = fight_dataset("test", n_fights)
    lines += ["", f"Fights: {len(specs)} from the test split.", "",
              "| Player | Win rate | Mean reward | Mean HP left (wins) |", "|---|---|---|---|"]
    for player in (RandomPlayer(0), GreedyPlayer(), PlannerPlayer(), TwoTurnPlanner()):
        results = [run_fight(s, player) for s in specs]
        wins = [r for r in results if r.won]
        hp = sum(r.hp_left for r in wins) / len(wins) if wins else 0
        lines.append(f"| {player.name} | {len(wins) / len(results):.0%} | "
                     f"{sum(r.reward for r in results) / len(results):.3f} | {hp:.1f} |")
    n_level = max(1, n_fights // 2)
    levels = range(MAX_LEVEL + 1)
    lines += ["", f"Fight levels (each adds +15% enemy HP and +1 damage per hit), "
                  f"win rate on {n_level} test fights:", "",
              "| Player | " + " | ".join(f"level {k}" for k in levels) + " |",
              "|---" * (len(levels) + 1) + "|"]
    for player in (GreedyPlayer(), PlannerPlayer()):
        cells = []
        for k in levels:
            results = [run_fight(s, player) for s in fight_dataset("test", n_level, level=k)]
            cells.append(f"{sum(r.won for r in results) / len(results):.0%}")
        lines.append(f"| {player.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _benchmark(args):
    print(benchmark(*(int(a) for a in args[:2])))


def _examples(args):
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "puzzles_sample.jsonl", "w", encoding="utf-8") as f:
        for difficulty in ("easy", "medium", "hard"):
            for p in dataset("train", 3, difficulty):
                f.write(json.dumps({**p.to_dict(), "prompt": p.prompt()}) + "\n")
    hard = dataset("train", 1, "hard")[0]
    greedy = greedy_line(hard)
    g = score_line(hard, greedy)
    with open(out / "puzzle_walkthrough.md", "w", encoding="utf-8") as f:
        f.write(f"# A hard puzzle ({hard.id})\n\n```\n{hard.prompt()}\n```\n\n"
                f"**Exact answer:** {', '.join(hard.solution)}  \n"
                f"Only {hard.winning_lines} of {hard.total_lines} possible lines win.\n\n"
                f"**Greedy player:** {', '.join(greedy) or '(nothing)'} -> {g.damage}/{hard.enemy_hp} "
                f"damage, not enough.\n")
    # A fight the planner wins and the greedy player loses shows what the task rewards.
    spec = next(s for s in fight_dataset("train", 100)
                if run_fight(s, PlannerPlayer()).won and not run_fight(s, GreedyPlayer()).won)
    for player in (PlannerPlayer(), GreedyPlayer()):
        record: list[str] = []
        r = run_fight(spec, player, record)
        with open(out / f"fight_transcript_{player.name}.txt", "w", encoding="utf-8") as f:
            f.write(f"# {player.name} player, fight seed {r.seed}: won={r.won} reward={r.reward:.3f}\n\n")
            f.write("\n\n".join(record) + "\n")
    print(f"wrote examples to {out}")


def _export(args):
    out = Path(args[0])
    n_train = int(args[1]) if len(args) > 1 else 2000
    n_test = int(args[2]) if len(args) > 2 else 300
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("test", n_test)):
        with open(out / f"puzzles_{split}.jsonl", "w", encoding="utf-8") as f:
            for p in dataset(split, n):
                f.write(json.dumps({"id": p.id, "difficulty": p.difficulty, "prompt": p.prompt(),
                                    "answer": ", ".join(p.solution), "puzzle": p.to_dict()}) + "\n")
        with open(out / f"fights_{split}.jsonl", "w", encoding="utf-8") as f:
            for spec in fight_dataset(split, n):
                f.write(json.dumps({"id": spec.id, "seed": spec.seed, "enemy": spec.enemy,
                                    "deck": list(spec.deck), "prompt": FightEnv(spec).intro()}) + "\n")
    print(f"wrote {n_train} train and {n_test} test puzzles and fights to {out}")


def rules_markdown() -> str:
    """docs/RULES.md, generated from the engine's own rules text and data."""
    from .cards import CARDS
    from .enemies import ENEMIES, describe_intent
    from .text import RULES

    rules = "\n".join(RULES.splitlines()[1:])
    cards = "\n".join(f"| {c.name} | {c.cost} | {c.kind} | {c.text}{' Exhaust.' if c.exhaust else ''} |"
                      for c in CARDS.values())
    enemies = "\n".join(
        f"| {e.name} | {e.max_hp} | {e.start_shield} | "
        + " &rarr; ".join(describe_intent(i) for i in e.pattern) + " |"
        for e in ENEMIES.values())
    return f"""# Card Duel rules

This file is generated by `python -m card_duel rules docs/RULES.md` from the engine's own
text and data. A test checks that it stays in sync.

## Rules

{rules}

## Cards

| Card | Cost | Type | Effect |
|---|---|---|---|
{cards}

## Enemies

Each enemy repeats its intents in this order, starting at a seeded point in the cycle.

| Enemy | HP | Starting Shield | Intent cycle |
|---|---|---|---|
{enemies}
"""


def _rules(args):
    text = rules_markdown()
    if args:
        Path(args[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(args[0]).write_text(text, encoding="utf-8")
    else:
        print(text)


def _export_sft(args):
    from .sft import fight_example, puzzle_example

    out = Path(args[0])
    n_puzzles = int(args[1]) if len(args) > 1 else 1000
    n_fights = int(args[2]) if len(args) > 2 else 200
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sft_puzzles.jsonl", "w", encoding="utf-8") as f:
        for p in dataset("train", n_puzzles):
            f.write(json.dumps(puzzle_example(p)) + "\n")
    kept = 0
    with open(out / "sft_fights.jsonl", "w", encoding="utf-8") as f:
        for spec in fight_dataset("train", n_fights):
            ex = fight_example(spec)
            if ex["won"]:  # keep only winning games as targets
                f.write(json.dumps(ex) + "\n")
                kept += 1
    print(f"wrote {n_puzzles} puzzle solutions and {kept}/{n_fights} winning fights (train split) to {out}")


def _eval_model(args):
    import argparse

    from .llm_eval import ChatClient, eval_fights, eval_puzzles, summarize

    ap = argparse.ArgumentParser(prog="python -m card_duel eval-model",
                                 description="Score an OpenAI-compatible chat model. Reads OPENAI_API_KEY "
                                             "and OPENAI_BASE_URL from the environment.")
    ap.add_argument("model")
    ap.add_argument("--task", choices=["puzzle", "fight"], default="puzzle")
    ap.add_argument("--n", type=int, default=50, help="number of tasks (default 50)")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--base-url", help="API base URL, e.g. https://api.openai.com/v1")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--window", type=int, default=16, help="fight messages kept after the opening (default 16)")
    ap.add_argument("--level", type=int, default=0, choices=range(MAX_LEVEL + 1),
                    help="fight difficulty level 0-3 (default 0)")
    ap.add_argument("--out", help="JSONL file for per-task results")
    a = ap.parse_args(args)
    if not os.environ.get("OPENAI_API_KEY"):
        print("note: OPENAI_API_KEY is not set; requests are sent without a key")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", a.model)
    out = a.out or f"outputs/{safe}-{a.task}-{a.split}.jsonl"
    client = ChatClient(a.model, base_url=a.base_url, temperature=a.temperature, max_tokens=a.max_tokens)
    if a.task == "puzzle":
        records = eval_puzzles(client, dataset(a.split, a.n, a.difficulty), a.workers, out)
    else:
        records = eval_fights(client, fight_dataset(a.split, a.n, level=a.level), a.workers, out, a.window)
    print(summarize(records))
    print(f"{client.calls} API calls; per-task details in {out}")


COMMANDS = {"puzzle": _puzzle, "solve": _solve, "check": _check, "fight": _fight,
            "benchmark": _benchmark, "examples": _examples, "export": _export,
            "export-sft": _export_sft, "eval-model": _eval_model, "rules": _rules}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 1
    COMMANDS[argv[0]](argv[1:])
    return 0


def _entry() -> None:
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    _entry()
