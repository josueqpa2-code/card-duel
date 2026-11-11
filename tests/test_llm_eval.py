"""Runs the model evaluator against a fake chat server on 127.0.0.1."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from card_duel.baselines import PlannerPlayer, run_fight
from card_duel.fight import FightEnv, fight_dataset
from card_duel.llm_eval import ChatClient, eval_fights, eval_puzzles, summarize
from card_duel.puzzles import dataset


class FakeChat(BaseHTTPRequestHandler):
    answers: dict = {}      # puzzle prompt -> reply text
    fail_first: int = 0     # how many requests get HTTP 429 first
    requests: list = []
    mirrors: dict = {}      # fight intro -> (FightEnv, PlannerPlayer)

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if cls.fail_first > 0:
            cls.fail_first -= 1
            self.send_response(429)
            self.end_headers()
            return
        msgs = body["messages"]
        first_user = msgs[1]["content"]
        if first_user in cls.answers:
            reply = cls.answers[first_user]
        elif first_user.startswith("You are playing Card Duel") and "FIGHT" in first_user:
            env, player = cls.mirrors[first_user]
            if msgs[-1]["role"] == "user" and len(msgs) > 2:
                # The evaluator must forward exactly what the engine said.
                assert msgs[-1]["content"] == env._last_obs, "observation mismatch"
            action = player.act(env)
            env._last_obs, *_ = env.step(action)
            reply = f"Let me think. <action>{action}</action>"
        else:
            reply = "no idea"
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LlmEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeChat.requests = []
        FakeChat.fail_first = 0

    def client(self, **kw):
        return ChatClient("fake-model", base_url=self.url, api_key="test-key", backoff=0.01, **kw)

    def test_puzzles_scored_and_logged(self):
        puzzles = dataset("test", 6)
        FakeChat.answers = {p.prompt(): f"Reasoning...\n<answer>{', '.join(p.solution)}</answer>" for p in puzzles[:4]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "puzzles.jsonl"
            records = eval_puzzles(self.client(), puzzles, workers=3, out=out)
            logged = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["solved"] for r in records], [True] * 4 + [False] * 2)
        self.assertEqual(len(logged), 6)
        req = FakeChat.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["auth"], "Bearer test-key")
        self.assertEqual(req["body"]["model"], "fake-model")
        self.assertEqual(req["body"]["messages"][0]["role"], "system")
        self.assertNotIn("temperature", req["body"])
        self.assertIn("Puzzles solved: 4/6", summarize(records))

    def test_retries_after_rate_limit(self):
        p = dataset("test", 1)[0]
        FakeChat.answers = {p.prompt(): f"<answer>{', '.join(p.solution)}</answer>"}
        FakeChat.fail_first = 2
        records = eval_puzzles(self.client(), [p], workers=1)
        self.assertTrue(records[0]["solved"])
        self.assertEqual(len(FakeChat.requests), 3)

    def test_gives_up_and_records_api_error(self):
        p = dataset("test", 1)[0]
        FakeChat.fail_first = 10
        records = eval_puzzles(self.client(retries=1), [p], workers=1)
        self.assertFalse(records[0]["solved"])
        self.assertTrue(records[0]["error"].startswith("api: HTTP 429"))

    def test_fights_follow_the_engine(self):
        specs = fight_dataset("test", 2)
        FakeChat.mirrors = {}
        for spec in specs:
            env = FightEnv(spec)
            env._last_obs = None
            FakeChat.mirrors[env.intro()] = (env, PlannerPlayer())
        records = eval_fights(self.client(), specs, workers=2, window=6)
        for spec, record in zip(specs, records):
            direct = run_fight(spec, PlannerPlayer())
            self.assertAlmostEqual(record["reward"], direct.reward)
            self.assertEqual(record["won"], direct.won)
            self.assertEqual(record["illegal_actions"], 0)
            self.assertTrue(record["finished"])
        for req in FakeChat.requests:
            roles = [m["role"] for m in req["body"]["messages"]]
            self.assertEqual(roles[:2], ["system", "user"])
            self.assertLessEqual(len(roles), 2 + 6)
            for a, b in zip(roles[2::2], roles[3::2]):
                self.assertEqual((a, b), ("assistant", "user"))
        self.assertIn("Fights won:", summarize(records))


if __name__ == "__main__":
    unittest.main()
