import json
import unittest

from card_duel.puzzles import (PROBE_HP, Puzzle, damage_value, dataset, generate, greedy_line,
                               lookahead_line, random_line, score_answer, score_line, _signature)
from card_duel.solver import search_turn

SEEDS = range(80)


def handmade(**kw) -> Puzzle:
    # Wind Up must come before Flurry: 3 x (2+4) + 6 = 24. Any other order deals at most 16.
    base = dict(id="hand-1", seed=-1, split="train", difficulty="medium", player_hp=20, energy=2,
                might=0, hand=("Wind Up", "Flurry", "Jab"), draw_pile=(), enemy_name="Bramble Golem",
                enemy_hp=24, enemy_max_hp=80, enemy_shield=0, enemy_burn=0, enemy_cracked=0,
                solution=("Wind Up", "Flurry", "Jab"), winning_lines=0, total_lines=0)
    base.update(kw)
    return Puzzle(**base)


class GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puzzles = [generate(s) for s in SEEDS]

    def test_deterministic(self):
        self.assertEqual(generate(7), generate(7))
        self.assertEqual(generate(7).prompt(), generate(7).prompt())

    def test_every_solution_wins(self):
        for p in self.puzzles:
            with self.subTest(seed=p.seed):
                r = score_line(p, p.solution)
                self.assertTrue(r.solved, r)
                self.assertEqual(r.reward, 1.0)

    def test_enemy_hp_is_exactly_the_best_possible_damage(self):
        for p in self.puzzles[:30]:
            with self.subTest(seed=p.seed):
                best = search_turn(p.battle(enemy_hp=PROBE_HP), damage_value(PROBE_HP))
                self.assertEqual(int(best.value), p.enemy_hp)

    def test_line_counts_are_consistent(self):
        for p in self.puzzles:
            self.assertGreaterEqual(p.winning_lines, 1)
            self.assertLessEqual(p.winning_lines * 2, p.total_lines)
            self.assertGreaterEqual(len(p.solution), 2)

    def test_difficulty_labels(self):
        for p in self.puzzles:
            greedy_wins = score_line(p, greedy_line(p)).solved
            self.assertEqual(p.difficulty == "easy", greedy_wins, p.seed)
        self.assertEqual({p.difficulty for p in self.puzzles}, {"easy", "medium", "hard"})

    def test_no_puzzle_is_in_both_splits(self):
        sigs = {"train": set(), "test": set()}
        for p in [generate(s) for s in range(300)]:
            c = dict(hand=p.hand, draw_pile=list(p.draw_pile), energy=p.energy, might=p.might,
                     player_hp=p.player_hp, enemy_hp=p.enemy_hp, enemy_shield=p.enemy_shield,
                     enemy_burn=p.enemy_burn, enemy_cracked=p.enemy_cracked)
            sigs[p.split].add(_signature(c))
        self.assertTrue(sigs["test"])
        self.assertFalse(sigs["train"] & sigs["test"])

    def test_dataset_filters(self):
        ps = dataset("test", 5, difficulty="hard")
        self.assertEqual(len(ps), 5)
        self.assertTrue(all(p.split == "test" and p.difficulty == "hard" for p in ps))

    def test_round_trip_through_json(self):
        p = self.puzzles[3]
        self.assertEqual(Puzzle.from_dict(json.loads(json.dumps(p.to_dict()))), p)

    def test_baseline_lines_are_legal(self):
        for p in self.puzzles:
            self.assertTrue(score_line(p, greedy_line(p)).legal)
            self.assertTrue(score_line(p, random_line(p, 1)).legal)
            self.assertTrue(score_line(p, lookahead_line(p, 2)).legal)

    def test_full_depth_lookahead_always_solves(self):
        small = [p for p in self.puzzles if len(p.hand) + len(p.draw_pile) <= 6]
        self.assertGreater(len(small), 5)
        for p in small:
            depth = len(p.hand) + len(p.draw_pile)
            self.assertTrue(score_line(p, lookahead_line(p, depth)).solved, p.seed)

    def test_prompt_shows_the_state_and_every_card(self):
        p = self.puzzles[0]
        text = p.prompt()
        self.assertIn(f"HP {p.enemy_hp}/{p.enemy_max_hp}", text)
        for name in set(p.hand) | set(p.draw_pile):
            self.assertIn(f"- {name} [", text)
        self.assertNotIn(", ".join(p.solution) + "</answer>", text.replace("Example: <answer>Wind Up, Jab, Jab", ""))


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.p = handmade()

    def check(self, completion, solved, legal=True):
        r = score_answer(self.p, completion)
        self.assertEqual((r.solved, r.legal), (solved, legal), r)
        return r

    def test_correct_order(self):
        self.check("<answer>Wind Up, Flurry, Jab</answer>", True)

    def test_messy_formatting(self):
        self.check("Plan:\n<answer>\n1. wind up\n2. FLURRY\n3. play jab.\n</answer>", True)
        self.check("<answer>WindUp -> Flurry -> Jab</answer>", True)

    def test_bare_answer_without_tags(self):
        self.check("Jab, Wind Up, Flurry", True)

    def test_wrong_order_loses(self):
        r = self.check("<answer>Wind Up, Jab, Flurry</answer>", False)
        self.assertEqual(r.damage, 16)
        self.assertEqual(r.reward, 0.0)
        partial = score_answer(self.p, "<answer>Wind Up, Jab, Flurry</answer>", partial_credit=True)
        self.assertAlmostEqual(partial.reward, 0.5 * 16 / 24)

    def test_cards_after_the_win_are_ignored(self):
        r = self.check("<answer>Wind Up, Flurry, Jab, Jab, Bulwark</answer>", True)
        self.assertEqual(r.played, ("Wind Up", "Flurry", "Jab"))

    def test_illegal_play_scores_zero(self):
        r = self.check("<answer>Bulwark</answer>", False, legal=False)
        self.assertEqual(r.reward, 0.0)
        self.assertIn("not in your hand", r.error)

    def test_unknown_card(self):
        r = self.check("<answer>Wind Up, Fireball</answer>", False, legal=False)
        self.assertIn("unknown card", r.error)

    def test_no_answer(self):
        r = self.check("I would play Wind Up first and then Flurry.", False, legal=False)
        self.assertIn("no <answer>", r.error)

    def test_last_answer_tag_counts(self):
        self.check("<answer>Jab</answer> wait, better: <answer>Wind Up, Flurry, Jab</answer>", True)

    def test_end_turn_stops_the_line(self):
        r = self.check("<answer>Wind Up, end turn, Flurry</answer>", False)
        self.assertEqual(r.played, ("Wind Up",))

    def test_self_defeat_scores_zero(self):
        p = handmade(player_hp=3, hand=("Overclock", "Jab", "Jab", "Jab"), energy=1, enemy_hp=18)
        r = score_answer(p, "<answer>Overclock, Jab, Jab, Jab</answer>")
        self.assertFalse(r.solved)
        self.assertEqual(r.reward, 0.0)
        self.assertIn("HP reached 0", r.error)


if __name__ == "__main__":
    unittest.main()
