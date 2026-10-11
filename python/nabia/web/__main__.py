"""python -m nabia.web --config ai_config.json [--host 127.0.0.1] [--port 8000]"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from ..ai.config import AIConfig
from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m nabia.web", description="nabia agents web app")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=Path(".nabia_ai"))
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: local only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(AIConfig.load(args.config), args.workdir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
