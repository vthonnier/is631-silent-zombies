"""Shared CLI helpers for the experiment scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow `python experiments/eX.py` without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmas.config import Config  # noqa: E402


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--limit", type=int, default=10, help="number of queries")
    p.add_argument("--out", default=None, help="override out_dir")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config)
    if args.out:
        cfg.out_dir = args.out
    return cfg


def dump(cfg: Config, name: str, payload: dict) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
