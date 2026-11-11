import unittest

from card_duel.baselines import PlannerPlayer
from card_duel.fight import FightEnv, fight_dataset
from card_duel.puzzles import dataset, score_answer
from card_duel.sft import fight_example, puzzle_example, puzzle_explanation


class SftTest(unittest.TestCase):
    def test_worked_solutions_score_full_marks(self):
        for p in dataset("train", 40):
            ex = puzzle_example(p)
            reply = ex["messages"][-1]["content"]
            self.assertEqual(ex["messages"][-1]["role"], "assistant")
            self.assertEqual(score_answer(p, reply).reward, 1.0, p.id)
            for name in p.solution:
                self.assertIn(name, reply)
            self.assertIn("defeated", reply)

    def test_explanation_numbers_add_up(self):
        p = dataset("train", 1, difficulty="hard")[0]
        text = puzzle_explanation(p)
        self.assertTrue(text.startswith(f"The enemy has {p.enemy_hp} HP"))
        self.assertIn("Enemy at 0 HP", text.replace("bringing it to 0 HP", "Enemy at 0 HP"))

    def test_fight_transcript_replays_exactly(self):
        spec = fight_dataset("train", 1)[0]
        ex = fight_example(spec, PlannerPlayer())
        msgs = ex["messages"]
        self.assertEqual([m["role"] for m in msgs[:2]], ["system", "user"])
        env = FightEnv(spec)
        self.assertEqual(msgs[1]["content"], env.intro())
        for i, m in enumerate(msgs[2:]):
            self.assertEqual(m["role"], "assistant" if i % 2 == 0 else "user")
        for i in range(2, len(msgs), 2):
            obs, *_ = env.step(msgs[i]["content"])
            if i + 1 < len(msgs):
                self.assertEqual(msgs[i + 1]["content"], obs)
        self.assertTrue(env.done)
        self.assertAlmostEqual(env.reward(), ex["reward"])
        self.assertEqual(msgs[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
