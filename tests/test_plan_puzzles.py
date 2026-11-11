import json
import unittest

from card_duel.plan_puzzles import (PlanPuzzle, generate, greedy_plan, myopic_plan, plan_dataset,
                                    score_plan, score_plan_answer, solve)

SEEDS = range(40)


def handmade(**kw) -> PlanPuzzle:
    # Turn 1: Guard blocks the 8-damage hit (HP 6 would not survive it), Ignite adds Burn.
    # Turn 2 draws Jab, Jab, Spark, Flurry, Spark.
    base = dict(id="plan-hand", seed=-1, split="train", difficulty="hard", player_hp=6, energy=2,
                might=0, hand=("Guard", "Ignite", "Jab"), draw_pile=("Jab", "Jab", "Spark", "Flurry", "Spark"),
                enemy_name="Bramble Golem", enemy_hp=1, enemy_max_hp=80, enemy_shield=0, enemy_burn=0,
                enemy_cracked=0, intent=(("attack", 8, 1),), solution=((), ()))
    base.update(kw)
    p = PlanPuzzle(**base)
    value, t1, t2 = solve(p.battle(enemy_hp=10**6), p.intent)
    return PlanPuzzle(**{**base, "enemy_hp": int(value), "solution": (t1, t2)})


class PlanGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puzzles = [generate(s) for s in SEEDS]

    def test_deterministic(self):
        self.assertEqual(generate(3), generate(3))

    def test_solutions_win_and_need_two_turns(self):
        for p in self.puzzles:
            with self.subTest(seed=p.seed):
                r = score_plan(p, *p.solution)
                self.assertTrue(r.solved, r)
                self.assertTrue(p.solution[1], "a second turn must be needed")

    def test_enemy_hp_is_the_best_possible(self):
        for p in self.puzzles[:15]:
            value, _, _ = solve(p.battle(enemy_hp=10**6), p.intent)
            self.assertEqual(int(value), p.enemy_hp, p.seed)

    def test_difficulty_matches_the_myopic_player(self):
        for p in self.puzzles:
            self.assertEqual(p.difficulty == "medium", score_plan(p, *myopic_plan(p)).solved, p.seed)
        self.assertEqual({p.difficulty for p in self.puzzles}, {"medium", "hard"})

    def test_baselines_are_legal(self):
        for p in self.puzzles:
            self.assertTrue(score_plan(p, *greedy_plan(p)).legal)
            self.assertTrue(score_plan(p, *myopic_plan(p)).legal)

    def test_json_round_trip(self):
        p = self.puzzles[5]
        self.assertEqual(PlanPuzzle.from_dict(json.loads(json.dumps(p.to_dict()))), p)

    def test_dataset_split_filter(self):
        ps = plan_dataset("test", 4, difficulty="hard")
        self.assertTrue(all(p.split == "test" and p.difficulty == "hard" for p in ps))

    def test_prompt(self):
        p = self.puzzles[0]
        text = p.prompt()
        self.assertIn("WIN WITHIN TWO TURNS", text)
        self.assertIn(", ".join(p.draw_pile), text)
        for name in set(p.hand) | set(p.draw_pile):
            self.assertIn(f"- {name} [", text)


class PlanScoringTest(unittest.TestCase):
    def setUp(self):
        self.p = handmade()

    def test_handmade_needs_the_guard(self):
        self.assertIn("Guard", self.p.solution[0])
        r = score_plan(self.p, ["Ignite", "Jab"], list(self.p.solution[1]))
        self.assertFalse(r.solved)
        self.assertEqual(r.error, "your HP reached 0")

    def test_answer_parsing(self):
        t1, t2 = (", ".join(t) for t in self.p.solution)
        self.assertTrue(score_plan_answer(self.p, f"<answer>{t1} | {t2}</answer>").solved)
        self.assertTrue(score_plan_answer(self.p, f"plan...\n<answer>\n{t1}\n|\n{t2}\n</answer>").solved)
        self.assertFalse(score_plan_answer(self.p, f"<answer>{t1}</answer>").solved)
        r = score_plan_answer(self.p, f"<answer>{t1} | {t2} | Jab</answer>")
        self.assertIn("one '|'", r.error)
        self.assertIn("no <answer>", score_plan_answer(self.p, f"{t1} | {t2}").error)

    def test_illegal_second_turn(self):
        r = score_plan(self.p, list(self.p.solution[0]), ["Bulwark"])
        self.assertFalse(r.legal)
        self.assertIn("turn 2", r.error)


if __name__ == "__main__":
    unittest.main()
