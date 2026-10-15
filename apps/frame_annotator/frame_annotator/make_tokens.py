"""Create per-annotator secret tokens.

    python -m frame_annotator.make_tokens ana bruno carla --out annotators.json

The JSON file maps token -> annotator name; load it as the Fly secret
ANNOTATOR_TOKENS_JSON. Invite links are ``https://<app>/#token=<token>``.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys

from .labels import is_safe_id


def make_tokens(names) -> dict:
    seen = set()
    for name in names:
        if not is_safe_id(name):
            raise ValueError(f"unsafe annotator name {name!r}")
        if name in seen:
            raise ValueError(f"duplicate annotator {name!r}")
        seen.add(name)
    return {secrets.token_urlsafe(32): name for name in names}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--out", required=True, help="file to create (never overwritten)")
    args = parser.parse_args(argv)
    tokens = make_tokens(args.names)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(tokens, handle, indent=2)
    for token, name in tokens.items():
        print(f"{name}\t#token={token}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
