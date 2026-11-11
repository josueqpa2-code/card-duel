import unittest

from card_duel.engine import (Battle, Fighter, IllegalAction, card_error, draw, end_of_turn, enemy_turn,
                              incoming_damage, play_card, start_of_turn)
from card_duel.rng import Rng, mix


def battle(hand, energy=3, might=0, charge=0, enemy_hp=100, shield=0, burn=0, cracked=0,
           draw_pile=(), discard=(), player_hp=40):
    return Battle(
        player=Fighter("You", player_hp, 40, might=might),
        enemy=Fighter("Dummy", enemy_hp, enemy_hp, shield=shield, burn=burn, cracked=cracked),
        energy=energy, hand=list(hand), draw_pile=list(draw_pile), discard=list(discard),
        charge=charge, seed=123,
    )


class RngTest(unittest.TestCase):
    def test_splitmix64_reference_value(self):
        # Published first output of SplitMix64 seeded with 0.
        self.assertEqual(Rng(0).next_u64(), 0xE220A8397B1DCDAF)

    def test_same_seed_same_sequence(self):
        a, b = Rng(mix(1, 2, 3)), Rng(mix(1, 2, 3))
        self.assertEqual([a.below(1000) for _ in range(20)], [b.below(1000) for _ in range(20)])

    def test_shuffle_is_a_permutation(self):
        items = list(range(30))
        Rng(9).shuffle(items)
        self.assertEqual(sorted(items), list(range(30)))
        self.assertNotEqual(items, list(range(30)))


class DamageTest(unittest.TestCase):
    def test_full_damage_pipeline(self):
        b = battle(["Jab"], might=1, charge=4, shield=5, cracked=1)
        play_card(b, "Jab")
        # (6 + 1 Might + 4 Charge) = 11, Cracked -> 16, Shield absorbs 5 -> 11 HP lost.
        self.assertEqual(b.enemy.hp, 89)
        self.assertEqual(b.enemy.shield, 0)
        self.assertEqual(b.charge, 0)

    def test_charge_applies_to_every_hit_then_is_used_up(self):
        b = battle(["Wind Up", "Flurry", "Jab"])
        play_card(b, "Wind Up")
        play_card(b, "Flurry")
        self.assertEqual(b.enemy.hp, 100 - 3 * 6)
        play_card(b, "Jab")
        self.assertEqual(b.enemy.hp, 100 - 18 - 6)

    def test_charge_carries_past_skills(self):
        b = battle(["Wind Up", "Guard", "Jab"])
        play_card(b, "Wind Up")
        play_card(b, "Guard")
        play_card(b, "Jab")
        self.assertEqual(b.enemy.hp, 90)

    def test_finisher_counts_only_earlier_attacks(self):
        b = battle(["Jab", "Spark", "Finisher"], energy=4)
        play_card(b, "Jab")
        play_card(b, "Spark")
        play_card(b, "Finisher")
        self.assertEqual(b.enemy.hp, 100 - 6 - 3 - (4 + 3 * 2))

    def test_cinder_strike_adds_burn(self):
        b = battle(["Cinder Strike"], burn=4)
        play_card(b, "Cinder Strike")
        self.assertEqual(b.enemy.hp, 92)

    def test_shatter_removes_shield_first(self):
        b = battle(["Shatter"], shield=10)
        play_card(b, "Shatter")
        self.assertEqual((b.enemy.shield, b.enemy.hp), (0, 97))

    def test_crack_hits_before_cracking(self):
        b = battle(["Crack", "Jab"])
        play_card(b, "Crack")
        self.assertEqual(b.enemy.hp, 93)
        self.assertEqual(b.enemy.cracked, 2)
        play_card(b, "Jab")
        self.assertEqual(b.enemy.hp, 93 - 9)

    def test_stoke_doubles_burn_and_exhausts(self):
        b = battle(["Ignite", "Stoke"])
        play_card(b, "Ignite")
        play_card(b, "Stoke")
        self.assertEqual(b.enemy.burn, 6)
        self.assertEqual(b.exhausted, ["Stoke"])
        self.assertEqual(b.discard, ["Ignite"])

    def test_hits_stop_once_enemy_is_dead(self):
        b = battle(["Flurry"], enemy_hp=3)
        play_card(b, "Flurry")
        self.assertEqual(b.enemy.hp, -1)  # two hits landed, the third was skipped
        self.assertEqual(b.outcome(), "win")


class UpkeepTest(unittest.TestCase):
    def test_burn_ignores_shield_and_ticks_down(self):
        f = Fighter("X", 20, 20, shield=10, burn=3)
        lost = start_of_turn(f)
        self.assertEqual((lost, f.hp, f.burn, f.shield), (3, 17, 2, 0))

    def test_cracked_ticks_at_end_of_turn(self):
        f = Fighter("X", 20, 20, cracked=2)
        end_of_turn(f)
        self.assertEqual(f.cracked, 1)

    def test_enemy_intent_runs_in_order(self):
        b = battle([], player_hp=40)
        enemy_turn(b, (("crack", 2), ("attack", 10, 1)))
        self.assertEqual(b.player.hp, 25)  # 10 x1.5 because the Crack came first
        self.assertEqual(b.player.cracked, 2)

    def test_incoming_damage_matches_the_real_hit(self):
        b = battle([], player_hp=40)
        b.player.cracked = 1
        b.enemy.might = 2
        intent = (("attack", 5, 2),)
        expected = incoming_damage(b, intent)
        enemy_turn(b, intent)
        self.assertEqual(40 - b.player.hp, expected)
        self.assertEqual(expected, 2 * ((5 + 2) * 3 // 2))

    def test_enemy_burn_can_kill_before_it_acts(self):
        b = battle([], enemy_hp=3, burn=3, player_hp=40)
        enemy_turn(b, (("attack", 30, 1),))
        self.assertEqual(b.outcome(), "win")
        self.assertEqual(b.player.hp, 40)


class DrawAndLegalityTest(unittest.TestCase):
    def test_scout_draws_from_the_top_and_exhausts(self):
        b = battle(["Scout"], draw_pile=["Jab", "Crack", "Spark"])
        play_card(b, "Scout")
        self.assertEqual(sorted(b.hand), ["Crack", "Jab"])
        self.assertEqual(b.draw_pile, ["Spark"])
        self.assertEqual(b.exhausted, ["Scout"])

    def test_reshuffle_is_deterministic(self):
        def run():
            b = battle([], discard=["Jab", "Crack", "Spark", "Guard", "Flurry"])
            draw(b, 3)
            return b.hand, b.draw_pile
        self.assertEqual(run(), run())
        hand, pile = run()
        self.assertEqual(len(hand), 3)
        self.assertEqual(len(pile), 2)

    def test_hand_limit(self):
        b = battle(["Jab"] * 9 + ["Scout"], draw_pile=["Spark", "Spark", "Spark"])
        play_card(b, "Scout")
        self.assertEqual(len(b.hand), 10)

    def test_blind_copy_counts_draws_without_revealing_them(self):
        b = battle(["Scout"], draw_pile=["Jab", "Crack"])
        blind = b.blind_copy()
        play_card(blind, "Scout")
        self.assertEqual(blind.hand, [])
        self.assertEqual(blind.pending_draws, 2)
        self.assertEqual(b.draw_pile, ["Jab", "Crack"])  # the real game is untouched

    def test_overclock_can_defeat_you(self):
        b = battle(["Overclock"], player_hp=3)
        play_card(b, "Overclock")
        self.assertEqual(b.outcome(), "loss")
        self.assertEqual(b.energy, 5)

    def test_illegal_plays_raise(self):
        b = battle(["Crack"], energy=1)
        self.assertIn("costs 2", card_error(b, "Crack"))
        self.assertIn("not in your hand", card_error(b, "Jab"))
        self.assertIn("no card", card_error(b, "Fireball"))
        with self.assertRaises(IllegalAction):
            play_card(b, "Crack")
        self.assertEqual(b.hand, ["Crack"])
        self.assertEqual(b.energy, 1)


if __name__ == "__main__":
    unittest.main()
