"""Checks our side of the verifiers adapter with stand-in `verifiers` and
`datasets` modules. This proves the reward functions and the multi-turn loop
work; it does not prove compatibility with a real verifiers release."""

import asyncio
import json
import sys
import types
import unittest

from card_duel.baselines import PlannerPlayer, run_fight
from card_duel.fight import FightEnv


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


class _Rubric:
    def __init__(self, funcs, weights, **kw):
        self.funcs, self.weights = funcs, weights


class _Env:
    def __init__(self, **kw):
        self.kw = kw


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("verifiers", "datasets")}
        vf = types.ModuleType("verifiers")
        vf.Rubric, vf.SingleTurnEnv, vf.MultiTurnEnv = _Rubric, _Env, _Env
        ds = types.ModuleType("datasets")
        ds.Dataset = _Dataset
        sys.modules["verifiers"], sys.modules["datasets"] = vf, ds

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_puzzle_rewards(self):
        import card_duel
        env = card_duel.load_environment(task="puzzle", num_train=3, num_eval=2)
        rows = env.kw["dataset"].rows
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(env.kw["eval_dataset"].rows), 2)
        reward, solved, legal = env.kw["rubric"].funcs
        for row in rows:
            json.dumps(row)  # rows must be plain data
            good = [{"role": "assistant", "content": f"<answer>{row['answer']}</answer>"}]
            self.assertEqual(reward(completion=good, info=row["info"]), 1.0)
            self.assertEqual(solved(completion=good, info=row["info"]), 1.0)
            bad = [{"role": "assistant", "content": "<answer>Fireball</answer>"}]
            self.assertEqual(reward(completion=bad, info=row["info"]), 0.0)
            self.assertEqual(legal(completion=bad, info=row["info"]), 0.0)
            self.assertEqual(reward(completion=f"<answer>{row['answer']}</answer>", info=row["info"]), 1.0)

    def test_plan_rewards(self):
        import card_duel
        env = card_duel.load_environment(task="plan", num_train=2, num_eval=1)
        reward, solved, legal = env.kw["rubric"].funcs
        for row in env.kw["dataset"].rows:
            json.dumps(row)
            good = [{"role": "assistant", "content": f"<answer>{row['answer']}</answer>"}]
            self.assertEqual(reward(completion=good, info=row["info"]), 1.0)
            bad = [{"role": "assistant", "content": "<answer>Jab | Jab | Jab</answer>"}]
            self.assertEqual(legal(completion=bad, info=row["info"]), 0.0)

    def test_fight_rollout_matches_direct_play(self):
        import card_duel
        env = card_duel.load_environment(task="fight", num_train=2, num_eval=1)
        row = env.kw["dataset"].rows[0]
        state = {"info": row["info"]}
        messages = [{"role": "user", "content": row["question"]}]
        player = PlannerPlayer()
        mirror = FightEnv(int(row["info"]["seed"]))
        while not asyncio.run(env.is_completed(messages, state)):
            action = player.act(mirror)
            expected_obs, *_ = mirror.step(action)
            messages.append({"role": "assistant", "content": f"<action>{action}</action>"})
            reply, state = asyncio.run(env.env_response(messages, state))
            self.assertEqual(reply[0]["content"], expected_obs)
            messages += reply
        json.dumps(state)
        fight_reward, won, illegal = env.kw["rubric"].funcs
        direct = run_fight(int(row["info"]["seed"]), PlannerPlayer())
        self.assertAlmostEqual(fight_reward(state=state), direct.reward)
        self.assertEqual(won(state=state), float(direct.won))
        self.assertEqual(illegal(state=state), 0.0)


if __name__ == "__main__":
    unittest.main()
