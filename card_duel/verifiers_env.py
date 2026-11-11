"""Legacy (verifiers v0) adapter. For verifiers v1 use the packages in environments/.

    pip install "card-duel[verifiers]"
    import card_duel
    env = card_duel.load_environment(task="puzzle")   # or task="fight"

Status: written against verifiers 0.1.x but NOT run in this repository,
because verifiers is not installed here. The core package it wraps is fully
tested. Rollout state only holds plain JSON data (the fight is rebuilt by
replaying actions), so saving results works.
"""

from __future__ import annotations

from typing import Any

from .fight import FightEnv, fight_dataset, make_fight
from .puzzles import Puzzle, dataset, score_answer

PUZZLE_SYSTEM = ("Solve the Card Duel puzzle. Think it through, then give your final play order "
                 "inside <answer></answer>.")
FIGHT_SYSTEM = ("Play the Card Duel fight. Each reply must contain exactly one action inside "
                "<action></action>.")
ACTIONS_KEY = "card_duel_actions"


def _text(completion: Any) -> str:
    """Completion as text, for both chat (list of messages) and plain formats."""
    if isinstance(completion, str):
        return completion
    for message in reversed(completion or []):
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            return content if isinstance(content, str) else str(content)
    return ""


def _puzzle_rows(puzzles: list[Puzzle]) -> list[dict]:
    return [{"question": p.prompt(), "answer": ", ".join(p.solution), "task": "card-duel-puzzle",
             "info": {"puzzle": p.to_dict()}} for p in puzzles]


def load_puzzle_environment(num_train: int = 2000, num_eval: int = 200, difficulty: str | None = None,
                            partial_credit: bool = False, **kwargs):
    import verifiers as vf
    from datasets import Dataset

    train = Dataset.from_list(_puzzle_rows(dataset("train", num_train, difficulty)))
    test = Dataset.from_list(_puzzle_rows(dataset("test", num_eval, difficulty)))

    def _result(completion, info):
        return score_answer(Puzzle.from_dict(info["puzzle"]), _text(completion), partial_credit)

    def puzzle_reward(completion, info, **_) -> float:
        return _result(completion, info).reward

    def solved(completion, info, **_) -> float:
        return float(_result(completion, info).solved)

    def legal(completion, info, **_) -> float:
        return float(_result(completion, info).legal)

    rubric = vf.Rubric(funcs=[puzzle_reward, solved, legal], weights=[1.0, 0.0, 0.0])
    return vf.SingleTurnEnv(dataset=train, eval_dataset=test, system_prompt=PUZZLE_SYSTEM,
                            rubric=rubric, **kwargs)


def _replay_with_text(state: dict) -> tuple[FightEnv, str]:
    """Rebuild the fight from its seed and the actions so far, plus the last observation."""
    info = state["info"]
    env = FightEnv(make_fight(int(info["seed"]), int(info.get("level", 0))))
    text = env.intro()
    for action in state.get(ACTIONS_KEY, []):
        text, *_ = env.step(action)
    return env, text


def _replay(state: dict) -> FightEnv:
    return _replay_with_text(state)[0]


def _fight_rows(n: int, split: str, level: int = 0) -> list[dict]:
    return [{"question": FightEnv(spec).intro(), "answer": "", "task": "card-duel-fight",
             "info": {"seed": spec.seed, "level": spec.level}}
            for spec in fight_dataset(split, n, level=level)]


def load_fight_environment(num_train: int = 2000, num_eval: int = 200, max_turns: int = 150,
                           level: int = 0, **kwargs):
    import verifiers as vf
    from datasets import Dataset

    class CardDuelFightEnv(vf.MultiTurnEnv):
        async def is_completed(self, messages, state, **kw) -> bool:
            return _replay(state).done

        async def env_response(self, messages, state, **kw):
            state.setdefault(ACTIONS_KEY, []).append(_text(messages))
            _, reply = _replay_with_text(state)
            return [{"role": "user", "content": reply}], state

    def fight_reward(state, **_) -> float:
        return _replay(state).reward()

    def won(state, **_) -> float:
        return float(_replay(state).won)

    def illegal_actions(state, **_) -> float:
        return float(_replay(state).illegal)

    rubric = vf.Rubric(funcs=[fight_reward, won, illegal_actions], weights=[1.0, 0.0, 0.0])
    return CardDuelFightEnv(
        dataset=Dataset.from_list(_fight_rows(num_train, "train", level)),
        eval_dataset=Dataset.from_list(_fight_rows(num_eval, "test", level)),
        system_prompt=FIGHT_SYSTEM, rubric=rubric, max_turns=max_turns, **kwargs)


def load_environment(task: str = "puzzle", **kwargs):
    if task == "puzzle":
        return load_puzzle_environment(**kwargs)
    if task == "fight":
        return load_fight_environment(**kwargs)
    raise ValueError("task must be 'puzzle' or 'fight'")
