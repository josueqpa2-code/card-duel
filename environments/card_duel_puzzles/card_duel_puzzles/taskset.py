"""Card Duel puzzles as a verifiers v1 taskset.

Single turn: the model reads the puzzle and answers with a play order inside
<answer></answer>. The reward is 1.0 when that line defeats the enemy (the
enemy's HP is exactly the best damage possible, found by exhaustive search).

Modelled on the gsm8k example in the verifiers repository. Not yet run
against a real verifiers install.
"""

import json
from typing import Literal

import verifiers.v1 as vf

from card_duel.plan_puzzles import PlanPuzzle, plan_dataset, score_plan_answer
from card_duel.puzzles import Puzzle, dataset, score_answer
from card_duel.verifiers_env import PLAN_SYSTEM as PLAN_SYSTEM_PROMPT

SYSTEM_PROMPT = ("Solve the Card Duel puzzle. Think it through, then give your final play order "
                 "inside <answer></answer>.")


class CardDuelPuzzleConfig(vf.TasksetConfig):
    split: Literal["train", "test"] = "test"
    num_tasks: int = 200
    # "one-turn": win this turn. "two-turn": win within two turns (answer "turn 1 | turn 2").
    kind: Literal["one-turn", "two-turn"] = "one-turn"
    difficulty: Literal["easy", "medium", "hard"] | None = None  # two-turn has medium and hard
    partial_credit: bool = False


class CardDuelPuzzleData(vf.TaskData):
    puzzle: dict
    kind: str = "one-turn"
    partial_credit: bool = False


class CardDuelPuzzleTask(vf.Task[CardDuelPuzzleData]):
    def _result(self, trace):
        reply = trace.last_reply or ""
        if self.data.kind == "two-turn":
            return score_plan_answer(PlanPuzzle.from_dict(self.data.puzzle), reply, self.data.partial_credit)
        return score_answer(Puzzle.from_dict(self.data.puzzle), reply, self.data.partial_credit)

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return self._result(trace).reward

    @vf.metric
    async def legal(self, trace: vf.Trace) -> float:
        return float(self._result(trace).legal)

    @vf.metric
    async def damage_share(self, trace: vf.Trace) -> float:
        return max(0, self._result(trace).damage) / self.data.puzzle["enemy_hp"]


class CardDuelPuzzleTaskset(vf.Taskset[CardDuelPuzzleTask, CardDuelPuzzleConfig]):
    def load(self) -> list[CardDuelPuzzleTask]:
        cfg = self.config
        two_turn = cfg.kind == "two-turn"
        puzzles = (plan_dataset(cfg.split, cfg.num_tasks, cfg.difficulty) if two_turn
                   else dataset(cfg.split, cfg.num_tasks, cfg.difficulty))
        return [
            CardDuelPuzzleTask(
                CardDuelPuzzleData(
                    name=p.id,
                    prompt=p.prompt(),
                    system_prompt=PLAN_SYSTEM_PROMPT if two_turn else SYSTEM_PROMPT,
                    puzzle=json.loads(json.dumps(p.to_dict())),
                    kind=cfg.kind,
                    partial_credit=cfg.partial_credit,
                ),
                cfg.task,
            )
            for p in puzzles
        ]
