"""Run the annotator locally against a folder of recordings (no Google Drive).

    python -m clip_annotator.dev --dataset C:/Users/me/Desktop/SenseCV --data ./devdata

Builds ``<data>/index.sqlite`` from the dataset, stores annotations under
``<data>/annotations`` and creates one sign-in token per ``--annotator``
(kept in ``<data>/dev_tokens.json``). Prints the invite links.
"""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from .app import create_app
from .index_db import IndexDb
from .publish import build_local_index
from .store import LocalDirStore
from .videos import LocalVideoSource
from .wsgi import parse_tokens

TOKENS_NAME = "dev_tokens.json"


def _tokens(data: Path, names) -> dict:
    path = data / TOKENS_NAME
    tokens = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    have = set(tokens.values())
    for name in names:
        if name not in have:
            tokens[secrets.token_urlsafe(32)] = name
    path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return parse_tokens(json.dumps(tokens))


def make_dev_app(dataset, data, annotators=("dev",)):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    index = IndexDb(data / "index.sqlite")
    build_local_index(dataset, index)
    tokens = _tokens(data, annotators)
    app = create_app(index=index, store=LocalDirStore(data / "annotations"), tokens=tokens,
                     video_source=LocalVideoSource(dataset))
    return app, tokens, index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path("clip_annotator_devdata"))
    parser.add_argument("--annotator", action="append", help="repeatable; default: dev")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5100)
    args = parser.parse_args(argv)
    app, tokens, index = make_dev_app(args.dataset, args.data, args.annotator or ["dev"])
    print(f"{len(index.clips())} clips indexed from {args.dataset}")
    for token, name in tokens.items():
        print(f"{name}: http://{args.host}:{args.port}/#token={token}")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
