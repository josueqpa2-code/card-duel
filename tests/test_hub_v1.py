"""Checks the verifiers v1 packages in environments/ against a stand-in
`verifiers.v1` module. This proves our loading, scoring and turn loop work
with the API shape used by the verifiers examples (gsm8k, textarena); it
does not prove compatibility with a real verifiers release."""

import asyncio
import json
import sys
import types
import unittest
from pathlib import Path

from card_duel.baselines import PlannerPlayer, run_fight
from card_duel.fight import FightEnv

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("card_duel_puzzles", "card_duel_fights")


class _Sub:
    def __class_getitem__(cls, item):
        return cls


class _Fields:
    task = None

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Task(_Sub):
    def __init__(self, data, config=None):
        self.data, self.config = data, config


class _Taskset(_Sub):
    def __init__(self, config):
        self.config = config


class _Env(_Sub):
    def __init__(self, config=None):
        self.config = config


def _decorator(fn=None, **_):
    return fn if fn is not None else (lambda f: f)


def _stub_module():
    v1 = types.ModuleType("verifiers.v1")
    v1.TaskData = type("TaskData", (_Fields,), {})
    v1.TasksetConfig = type("TasksetConfig", (_Fields,), {})
    v1.EnvConfig = type("EnvConfig", (_Fields,), {})
    v1.AgentConfig = type("AgentConfig", (_Fields,), {})
    v1.Task, v1.Taskset, v1.Env = _Task, _Taskset, _Env
    v1.reward = v1.metric = _decorator
    v1.Trace = v1.Runtime = v1.State = v1.TaskConfig = object
    root = types.ModuleType("verifiers")
    root.v1 = v1
    return root, v1


class _Trace:
    def __init__(self):
        self.rewards, self.info = {}, {}

    def record_reward(self, name, value, weight):
        self.rewards[name] = (value, weight)


class _Interaction:
    """Plays the model's side with the planner, checking every observation."""

    def __init__(self, seed):
        self.mirror, self.player, self.trace = FightEnv(seed), PlannerPlayer(), _Trace()
        self.expected = None
        self.checked = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def turn(self, text=None):
        if text is not None:
            assert text == self.expected, "the env must send the engine's observation"
            self.checked += 1
        if self.mirror.done:
            return types.SimpleNamespace(last_reply="", terminated=True)
        action = self.player.act(self.mirror)
        self.expected, *_ = self.mirror.step(action)
        return types.SimpleNamespace(last_reply=f"<action>{action}</action>", terminated=False)


class HubV1Test(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("verifiers", "verifiers.v1")}
        root, v1 = _stub_module()
        sys.modules["verifiers"], sys.modules["verifiers.v1"] = root, v1
        for pkg in PACKAGES:
            sys.path.insert(0, str(ROOT / "environments" / pkg))

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for pkg in PACKAGES:
            sys.path.remove(str(ROOT / "environments" / pkg))
            for name in [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]:
                del sys.modules[name]

    def test_puzzle_taskset(self):
        from card_duel_puzzles import CardDuelPuzzleTaskset
        from card_duel_puzzles.taskset import CardDuelPuzzleConfig
        tasks = CardDuelPuzzleTaskset(CardDuelPuzzleConfig(split="test", num_tasks=3, difficulty="hard",
                                                           partial_credit=False)).load()
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            json.dumps(task.data.puzzle)
            self.assertIn("PUZZLE: WIN THIS TURN", task.data.prompt)
            answer = ", ".join(task.data.puzzle["solution"])
            good = types.SimpleNamespace(last_reply=f"thinking...<answer>{answer}</answer>")
            bad = types.SimpleNamespace(last_reply="<answer>Fireball</answer>")
            empty = types.SimpleNamespace(last_reply=None)
            self.assertEqual(asyncio.run(task.correct(good)), 1.0)
            self.assertEqual(asyncio.run(task.damage_share(good)), 1.0)
            self.assertEqual(asyncio.run(task.correct(bad)), 0.0)
            self.assertEqual(asyncio.run(task.legal(bad)), 0.0)
            self.assertEqual(asyncio.run(task.correct(empty)), 0.0)

    def test_two_turn_puzzle_taskset(self):
        from card_duel_puzzles import CardDuelPuzzleTaskset
        from card_duel_puzzles.taskset import CardDuelPuzzleConfig
        tasks = CardDuelPuzzleTaskset(CardDuelPuzzleConfig(split="test", num_tasks=2, kind="two-turn",
                                                           difficulty=None, partial_credit=False)).load()
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            self.assertIn("WIN WITHIN TWO TURNS", task.data.prompt)
            t1, t2 = (", ".join(t) for t in task.data.puzzle["solution"])
            good = types.SimpleNamespace(last_reply=f"<answer>{t1} | {t2}</answer>")
            self.assertEqual(asyncio.run(task.correct(good)), 1.0)
            only_first = types.SimpleNamespace(last_reply=f"<answer>{t1}</answer>")
            self.assertEqual(asyncio.run(task.correct(only_first)), 0.0)

    def test_fight_env_turn_loop(self):
        from card_duel_fights import CardDuelFightEnv, CardDuelFightTaskset
        from card_duel_fights.taskset import CardDuelFightConfig
        tasks = CardDuelFightTaskset(CardDuelFightConfig(split="test", num_tasks=3)).load()
        for task in tasks:
            seed = int(task.data.info["seed"])
            self.assertEqual(task.data.prompt, FightEnv(seed).intro())
            interaction = _Interaction(seed)
            agents = types.SimpleNamespace(player=types.SimpleNamespace(interaction=lambda t: interaction))
            asyncio.run(CardDuelFightEnv().run(task, agents))
            direct = run_fight(seed, PlannerPlayer())
            self.assertAlmostEqual(interaction.trace.rewards["fight_reward"][0], direct.reward)
            self.assertEqual(interaction.trace.info["card_duel"]["won"], direct.won)
            self.assertGreater(interaction.checked, 5)


if __name__ == "__main__":
    unittest.main()
