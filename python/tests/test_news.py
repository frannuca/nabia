"""python -m unittest discover -s tests (from python/): data layer and scoring, with a stub in place of the LLM."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datalayer import NewsDB, load_config
from scoring import Prompt, score

ARTICLE = {"headline": "Fed signals rate cuts as inflation cools", "body": "Officials see easing ahead.",
           "source": "CNBC", "url": "https://x/1", "published_at": "2026-10-01T12:00:00Z", "received_at": "2026-10-01T12:05:00Z"}


class NewsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cfg = load_config()
        self.cfg["db"] = str(Path(self.dir.name) / "test.db")

    def tearDown(self):
        self.dir.cleanup()

    def test_dedup(self):
        with NewsDB(self.cfg["db"]) as db:
            self.assertEqual(db.add_news([ARTICLE]), 1)
            self.assertEqual(db.add_news([ARTICLE, {**ARTICLE, "url": "https://y/2"}]), 0)  # same url; same headline and day
            self.assertEqual(db.add_news([{**ARTICLE, "url": None, "received_at": "2026-10-02T09:00:00Z"}]), 1)

    def test_factors(self):
        self.assertEqual(len(self.cfg["factors"]), 27)
        for f in self.cfg["factors"].values():
            self.assertTrue({"name", "group", "region", "positive"} <= set(f))
            for key, transform, _ in f.get("proxy", []):
                self.assertRegex(key, r"^(fred|price):")
                self.assertIn(transform, ("logret", "diff"))

    def test_parse(self):
        p = Prompt(self.cfg["factors"])
        raw = {"relevant": True, "scores": [{"factor": "POLICY_FED", "sentiment": 1.7, "confidence": 0.9},
                                            {"factor": "NOT_A_FACTOR", "sentiment": 0.5, "confidence": 0.5}]}
        self.assertEqual(p.parse(raw), (True, [("POLICY_FED", 1.0, 0.9)]))
        self.assertEqual(p.parse({"relevant": True, "scores": []}), (False, []))
        self.assertIn("POLICY_FED | Federal Reserve policy", p.system)

    def test_score_resumes(self):
        answer = {"relevant": True, "scores": [{"factor": "POLICY_FED", "sentiment": 0.6, "confidence": 0.8}]}
        with NewsDB(self.cfg["db"]) as db:
            db.add_news([ARTICLE])
        with mock.patch("scoring.llm.LLM.__call__", return_value=json.loads(json.dumps(answer))):
            self.assertEqual(score(self.cfg), 1)
            self.assertEqual(score(self.cfg), 0)  # already scored by this model and prompt
        with NewsDB(self.cfg["db"]) as db:
            [row] = db.scores()
        self.assertEqual((row["factor"], row["sentiment"], row["confidence"]), ("POLICY_FED", 0.6, 0.8))


if __name__ == "__main__":
    unittest.main()


class ApiTest(unittest.TestCase):
    def test_parallel_anthropic_with_rate_limit(self):
        """The Anthropic path end to end against a local stub of the Messages API: 8 workers, one 429 retried."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        calls = {"n": 0}

        class Stub(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls["n"] += 1
                if calls["n"] == 3:  # one rate limit, retried after Retry-After
                    self.send_response(429), self.send_header("retry-after", "0"), self.end_headers()
                    return
                assert self.headers["x-api-key"] == "test-key" and body["tool_choice"]["name"] == "emit"
                answer = {"relevant": True, "scores": [{"factor": "POLICY_FED", "sentiment": 0.5, "confidence": 0.7}]}
                out = json.dumps({"content": [{"type": "tool_use", "name": "emit", "input": answer}]}).encode()
                self.send_response(200), self.send_header("Content-Length", str(len(out))), self.end_headers()
                self.wfile.write(out)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config()
            cfg["db"] = str(Path(tmp) / "t.db")
            cfg["llm"].update(provider="anthropic", workers=8)
            cfg["llm"]["anthropic"].update(base_url=f"http://127.0.0.1:{server.server_port}", api_key_env="TEST_KEY")
            with NewsDB(cfg["db"]) as db:
                db.add_news([{**ARTICLE, "headline": f"Fed item {k}", "url": f"https://x/{k}"} for k in range(30)])
            with mock.patch.dict("os.environ", {"TEST_KEY": "test-key"}):
                self.assertEqual(score(cfg), 30)
            with NewsDB(cfg["db"]) as db:
                self.assertEqual(len(db.scores(model="anthropic:%")), 30)
        server.shutdown(), server.server_close()
        self.assertEqual(calls["n"], 31)
