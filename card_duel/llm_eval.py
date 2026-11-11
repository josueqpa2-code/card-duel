"""Score any OpenAI-compatible chat model on Card Duel. Standard library only.

    OPENAI_API_KEY=...  [OPENAI_BASE_URL=https://other-provider/v1]
    python -m card_duel eval-model MODEL --task puzzle --n 50
    python -m card_duel eval-model MODEL --task fight --n 20

Each finished task is appended to a JSONL file, so a long run can be
inspected while it goes. Nothing is sent anywhere unless you run this.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .fight import FightEnv, FightSpec
from .puzzles import Puzzle, score_answer
from .verifiers_env import FIGHT_SYSTEM, PUZZLE_SYSTEM

RETRY_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class ChatClient:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 300.0
    retries: int = 4
    backoff: float = 2.0
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self._lock = threading.Lock()

    def complete(self, messages: list[dict]) -> str:
        body: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                             headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode())
                with self._lock:
                    self.calls += 1
                return payload["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as err:
                if err.code in RETRY_CODES and attempt < self.retries:
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                detail = err.read()[:300].decode(errors="replace")
                raise RuntimeError(f"HTTP {err.code} from {self.base_url}: {detail}") from err
            except (urllib.error.URLError, TimeoutError) as err:
                if attempt < self.retries:
                    time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise RuntimeError(f"could not reach {self.base_url}: {err}") from err
        raise AssertionError("unreachable")


class _Log:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if self.path:
            with self.lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


def eval_puzzles(client: ChatClient, puzzles: list[Puzzle], workers: int = 4,
                 out: str | Path | None = None, partial_credit: bool = False) -> list[dict]:
    log = _Log(out)

    def one(p: Puzzle) -> dict:
        messages = [{"role": "system", "content": PUZZLE_SYSTEM}, {"role": "user", "content": p.prompt()}]
        try:
            reply = client.complete(messages)
        except RuntimeError as err:
            record = {"task": "puzzle", "id": p.id, "difficulty": p.difficulty, "reward": 0.0,
                      "solved": False, "legal": False, "error": f"api: {err}", "reply": None}
            log.write(record)
            return record
        r = score_answer(p, reply, partial_credit)
        record = {"task": "puzzle", "id": p.id, "difficulty": p.difficulty, "reward": r.reward,
                  "solved": r.solved, "legal": r.legal, "damage": r.damage, "enemy_hp": p.enemy_hp,
                  "played": list(r.played), "solution": list(p.solution), "error": r.error, "reply": reply}
        log.write(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, puzzles))


def eval_fights(client: ChatClient, specs: list[FightSpec], workers: int = 2,
                out: str | Path | None = None, window: int = 16) -> list[dict]:
    """Play each fight to the end. The model sees the system prompt, the opening
    message and the last `window` messages (every observation repeats the full
    state, so older turns are not needed)."""
    log = _Log(out)
    window += window % 2  # whole (assistant, user) pairs, so roles keep alternating

    def one(spec: FightSpec) -> dict:
        env = FightEnv(spec)
        head = [{"role": "system", "content": FIGHT_SYSTEM}, {"role": "user", "content": env.intro()}]
        tail: list[dict] = []
        actions: list[str] = []
        error = None
        while not env.done:
            try:
                reply = client.complete(head + (tail[-window:] if window else tail))
            except RuntimeError as err:
                error = f"api: {err}"
                break
            actions.append(reply)
            obs, *_ = env.step(reply)
            tail += [{"role": "assistant", "content": reply}, {"role": "user", "content": obs}]
        record = {"task": "fight", "id": spec.id, "enemy": spec.enemy, "reward": env.reward(),
                  "won": env.won, "finished": env.done, "turns": env.battle.turn,
                  "hp_left": max(env.battle.player.hp, 0), "illegal_actions": env.illegal,
                  "steps": env.steps, "error": error, "actions": actions}
        log.write(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, specs))


def summarize(records: list[dict]) -> str:
    if not records:
        return "no results"
    lines = []
    puzzles = [r for r in records if r["task"] == "puzzle"]
    fights = [r for r in records if r["task"] == "fight"]
    if puzzles:
        groups = defaultdict(list)
        for r in puzzles:
            groups[r["difficulty"]].append(r)
        parts = [f"{d} {sum(r['solved'] for r in g)}/{len(g)}" for d, g in sorted(groups.items())]
        legal = sum(r["legal"] for r in puzzles) / len(puzzles)
        api = sum(1 for r in puzzles if (r.get("error") or "").startswith("api:"))
        lines.append(f"Puzzles solved: {sum(r['solved'] for r in puzzles)}/{len(puzzles)} "
                     f"({', '.join(parts)}); legal answers {legal:.0%}; API errors {api}")
    if fights:
        wins = sum(r["won"] for r in fights)
        mean = sum(r["reward"] for r in fights) / len(fights)
        illegal = sum(r["illegal_actions"] for r in fights) / len(fights)
        api = sum(1 for r in fights if r.get("error"))
        lines.append(f"Fights won: {wins}/{len(fights)}; mean reward {mean:.3f}; "
                     f"illegal actions per fight {illegal:.1f}; API errors {api}")
    return "\n".join(lines)
