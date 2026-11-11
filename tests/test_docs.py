import unittest
from pathlib import Path

from card_duel.__main__ import rules_markdown
from card_duel.cards import CARDS
from card_duel.enemies import ENEMIES

ROOT = Path(__file__).resolve().parents[1]


class DocsTest(unittest.TestCase):
    def test_rules_file_is_in_sync(self):
        on_disk = (ROOT / "docs" / "RULES.md").read_text(encoding="utf-8")
        self.assertEqual(on_disk.replace("\r\n", "\n"), rules_markdown(),
                         "run: python -m card_duel rules docs/RULES.md")

    def test_rules_file_lists_everything(self):
        text = rules_markdown()
        for name in list(CARDS) + list(ENEMIES):
            self.assertIn(f"| {name} |", text)


if __name__ == "__main__":
    unittest.main()
