#!/usr/bin/env python3
"""Rewrite SK_RAG image paths to the local SK_RAG_OPD data directories."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "/home/sunyw/SK_RAG/data/BeaverTails-V/opsd_images/response_recommendation_stratified3000_seed42/current/": str(ROOT / "data/consistent_current") + "/",
    "/home/sunyw/SK_RAG/data/BeaverTails-V/opsd_images/response_recommendation_stratified3000_seed42/retrieved/": str(ROOT / "data/consistent_retrieved") + "/",
    "/home/sunyw/SK_RAG/data/BeaverTails-V/opsd_images/response_recommendation_controlled_3k/current/": str(ROOT / "data/controlled_current") + "/",
    "/home/sunyw/SK_RAG/data/BeaverTails-V/opsd_images/response_recommendation_controlled_3k/retrieved/": str(ROOT / "data/controlled_retrieved") + "/",
}


def rewrite(value):
    if isinstance(value, str):
        for old, new in REPLACEMENTS.items():
            if value.startswith(old):
                return new + value[len(old):]
        return value
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    return value


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: relocate_data_paths.py FILE [FILE ...]")
    for arg in sys.argv[1:]:
        path = Path(arg)
        temp = path.with_suffix(path.suffix + ".tmp")
        with path.open(encoding="utf-8") as src, temp.open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(json.dumps(rewrite(json.loads(line)), ensure_ascii=False) + "\n")
        temp.replace(path)
        print(f"rewrote {path}")


if __name__ == "__main__":
    main()
