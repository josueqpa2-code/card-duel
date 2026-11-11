import re
import unittest

from card_duel.baselines import GreedyPlayer, PlannerPlayer, RandomPlayer, TwoTurnPlanner, run_fight
from card_duel.fight import FightEnv, fight_dataset, make_fight
from card_duel.text import parse_action


class FightSpecTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(make_fight(11), make_fight(11))

    def test_dataset_split(self):
        test = fight_dataset("test", 10)
        train = fight_dataset("train", 10)
        self.assertTrue(all(s.split == "test" for s in test))
        self.assertFalse({s.seed for s in test} & {s.seed for s in train})


class FightEnvTest(unittest.TestCase):
    def test_same_actions_same_result(self):
        def play():
            env = FightEnv(5)
            player = PlannerPlayer()
            log = []
            while not env.done:
                a = player.act(env)
                obs, reward, done, info = env.step(a)
                log.append((a, obs, reward))
            return log
        self.assertEqual(play(), play())

    def test_illegal_action_changes_nothing(self):
        env = FightEnv(3)
        before = env.state_text()
        obs, reward, done, info = env.step("<action>play Fireball</action>")
        self.assertIn("Illegal action", obs)
        self.assertEqual(env.state_text(), before)
        self.assertEqual(env.illegal, 1)
        self.assertFalse(done)
        missing = next(n for n in ("Bulwark", "Hone", "Stoke", "Scout", "Spark") if n not in env.battle.hand)
        obs, *_ = env.step(f"play {missing}")
        self.assertIn("not in your hand", obs)
        self.assertEqual(env.state_text(), before)

    def test_draw_order_is_hidden(self):
        env = FightEnv(8)
        line = re.search(r"Draw pile \(\d+ cards?, order hidden\): (.*)", env.state_text()).group(1)
        names = [part.split(" x")[0] for part in line.split(", ")]
        self.assertEqual(names, sorted(names))

    def test_reward_ranges(self):
        for seed in range(25):
            for player in (RandomPlayer(seed), GreedyPlayer()):
                r = run_fight(seed, player)
                if r.won:
                    self.assertGreaterEqual(r.reward, 0.5)
                    self.assertLessEqual(r.reward, 1.0)
                else:
                    self.assertLess(r.reward, 0.4)
                    self.assertGreaterEqual(r.reward, 0.0)

    def test_turn_limit(self):
        env = FightEnv(2, max_turns=2)
        done = False
        while not done:
            _, reward, done, info = env.step("end turn")
        self.assertFalse(info["won"])
        self.assertLess(reward, 0.4)

    def test_step_limit(self):
        env = FightEnv(2, max_steps=3)
        for _ in range(3):
            _, _, done, info = env.step("dance")
        self.assertTrue(done)
        self.assertFalse(info["won"])

    def test_planner_beats_greedy_beats_random(self):
        seeds = range(60)
        wins = {p.name: sum(run_fight(s, p).won for s in seeds)
                for p in (RandomPlayer(0), GreedyPlayer(), PlannerPlayer())}
        self.assertGreater(wins["planner"], wins["greedy"])
        self.assertGreater(wins["greedy"], wins["random"])

    def test_intro_has_rules_deck_and_help(self):
        env = FightEnv(4)
        text = env.intro()
        self.assertIn("RULES", text)
        self.assertIn("<action>end turn</action>", text)
        for name in set(env.spec.deck):
            self.assertIn(f"- {name} [", text)


class LevelTest(unittest.TestCase):
    def test_level_zero_is_the_default(self):
        self.assertEqual(make_fight(9), make_fight(9, 0))
        self.assertEqual(FightEnv(9).intro(), FightEnv(make_fight(9, 0)).intro())

    def test_levels_scale_the_enemy(self):
        base = FightEnv(make_fight(9, 0)).battle.enemy
        for level in (1, 2, 3):
            spec = make_fight(9, level)
            enemy = FightEnv(spec).battle.enemy
            self.assertEqual(enemy.might, level)
            self.assertEqual(enemy.max_hp, round(base.max_hp * (1 + 0.15 * level)))
            self.assertEqual((spec.enemy, spec.deck, spec.split), (make_fight(9).enemy, make_fight(9).deck,
                                                                   make_fight(9).split))
            self.assertIn(f"-L{level}", spec.id)
        with self.assertRaises(ValueError):
            make_fight(9, 4)

    def test_higher_levels_are_harder(self):
        wins = [sum(run_fight(s, PlannerPlayer()).won for s in fight_dataset("test", 30, level=k))
                for k in (0, 3)]
        self.assertGreater(wins[0], wins[1])


class NoPeekingTest(unittest.TestCase):
    def test_players_ignore_the_hidden_draw_order(self):
        """Reordering the draw pile (same cards) must not change any player's choice."""
        for seed in range(12):
            for player_cls in (GreedyPlayer, PlannerPlayer, TwoTurnPlanner):
                env = FightEnv(seed)
                for _ in range(3):  # a few steps in, so piles and discards are mixed
                    if env.done:
                        break
                    choice = player_cls().act(env)
                    real_pile, real_seed = list(env.battle.draw_pile), env.battle.seed
                    env.battle.draw_pile = list(reversed(real_pile))
                    env.battle.seed = real_seed + 1
                    self.assertEqual(player_cls().act(env), choice, (seed, player_cls.__name__))
                    env.battle.draw_pile, env.battle.seed = real_pile, real_seed
                    env.step(choice)

    def test_two_turn_planner_is_legal_and_deterministic(self):
        a = run_fight(3, TwoTurnPlanner())
        b = run_fight(3, TwoTurnPlanner())
        self.assertEqual(a, b)
        self.assertEqual(a.illegal, 0)


class ParseActionTest(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(parse_action("<action>play Wind Up</action>")[0], ("play", "Wind Up"))
        self.assertEqual(parse_action("windup")[0], ("play", "Wind Up"))
        self.assertEqual(parse_action("thinking... <action>End Turn</action>")[0], ("end", None))
        self.assertEqual(parse_action("<action>a</action><action>play Jab</action>")[0], ("play", "Jab"))
        self.assertIsNotNone(parse_action("I will play a card")[1])


if __name__ == "__main__":
    unittest.main()
