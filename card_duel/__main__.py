"""Command line: python -m card_duel <command>

  puzzle SEED              print the prompt for a puzzle
  solve SEED               print the puzzle's answer and how rare winning lines are
  check SEED "A, B, C"     score a play order for a puzzle
  fight SEED               play a fight yourself in the terminal
  benchmark [P] [F]        baseline scores on P test puzzles and F test fights
  examples DIR             write sample tasks and a full fight transcript
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .baselines import GreedyPlayer, PlannerPlayer, RandomPlayer, run_fight
from .fight import FightEnv, fight_dataset
from .puzzles import dataset, generate, greedy_line, random_line, score_answer, score_line


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
    env = FightEnv(int(args[0]))
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
    for player in (RandomPlayer(0), GreedyPlayer(), PlannerPlayer()):
        results = [run_fight(s, player) for s in specs]
        wins = [r for r in results if r.won]
        hp = sum(r.hp_left for r in wins) / len(wins) if wins else 0
        lines.append(f"| {player.name} | {len(wins) / len(results):.0%} | "
                     f"{sum(r.reward for r in results) / len(results):.3f} | {hp:.1f} |")
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


COMMANDS = {"puzzle": _puzzle, "solve": _solve, "check": _check, "fight": _fight,
            "benchmark": _benchmark, "examples": _examples}


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
