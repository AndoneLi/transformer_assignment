# -*- coding: utf-8 -*-
"""
Append one experiment summary (with full fields) into results/ablations.csv

Usage (examples)
----------------
# LM baseline（自动从 config 读取结构字段）
python scripts/append_result.py --metrics results/lm/metrics.csv \
    --task lm --dataset tiny_shakespeare --tag LM-global-120 \
    --config configs/base_lm.yaml

# 快速实验，命令行覆盖结构字段
python scripts/append_result.py --metrics results/lm/metrics.csv \
    --task lm --dataset tiny_shakespeare --tag LM-fast \
    --d_model 256 --n_heads 4 --n_layers 4 --d_ff 1024 --max_len 256 \
    --attn_type local --rel_pos alibi --window_size 128 --max_steps 180 --seed 42

Notes
-----
- Run from REPO ROOT so that relative path "results/ablations.csv" is valid.
- If some fields are missing in config/CLI, we fill "NA".
"""
import argparse
import csv
import os
from typing import Any, Dict, Optional

import pandas as pd

try:
    import yaml  # optional but recommended
except Exception:
    yaml = None


ABLATIONS_PATH = os.path.join("results", "ablations.csv")


def safe_get(d: Dict[str, Any], *keys, default=None):
    """Nested get: safe_get(cfg, 'model', 'd_model', default=None)"""
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        print("[WARN] PyYAML not installed; skip reading config.")
        return {}
    if not os.path.isfile(path):
        print(f"[WARN] config file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] yaml.safe_load failed: {e}")
            return {}


def to_num(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return x
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return None


def collect_model_fields(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Try to read from config, then override with CLI if provided.
    Support both top-level and cfg['model']/cfg['train'] style.
    """
    # Try various nesting patterns
    # d_model etc. may be under cfg['model'] or at top level
    # max_steps, seed may be under cfg['train'] or top level
    fields = {}

    def get_with_fallback(key, *paths):
        # paths: list of tuple path candidates
        for p in paths:
            v = safe_get(cfg, *p)
            if v is not None:
                return v
        return None

    fields["d_model"] = get_with_fallback("d_model", ("model", "d_model"), ("d_model",))
    fields["n_heads"] = get_with_fallback("n_heads", ("model", "n_heads"), ("n_heads",))
    fields["n_layers"] = get_with_fallback("n_layers", ("model", "n_layers"), ("n_layers",))
    fields["d_ff"] = get_with_fallback("d_ff", ("model", "d_ff"), ("d_ff",))
    fields["max_len"] = get_with_fallback("max_len", ("model", "max_len"), ("max_len",))

    fields["attn_type"] = get_with_fallback("attn_type", ("model", "attn_type"), ("attn_type",))
    fields["rel_pos"] = get_with_fallback("rel_pos", ("model", "rel_pos"), ("rel_pos",))
    fields["window_size"] = get_with_fallback("window_size", ("model", "window_size"), ("window_size",))

    fields["max_steps"] = get_with_fallback("max_steps", ("train", "max_steps"), ("max_steps",))
    fields["seed"] = get_with_fallback("seed", ("train", "seed"), ("seed",))

    # CLI overrides (if provided)
    for k in ["d_model", "n_heads", "n_layers", "d_ff", "max_len",
              "attn_type", "rel_pos", "window_size", "max_steps", "seed"]:
        cli_v = getattr(args, k, None)
        if cli_v is not None:
            fields[k] = cli_v

    # Normalize numeric fields
    for k in ["d_model", "n_heads", "n_layers", "d_ff", "max_len", "window_size", "max_steps", "seed"]:
        fields[k] = to_num(fields.get(k))

    # Normalize strings
    for k in ["attn_type", "rel_pos"]:
        v = fields.get(k)
        if v is not None:
            fields[k] = str(v)

    return fields


def read_metrics_tail(metrics_path: str) -> Dict[str, Any]:
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    if len(df) == 0:
        raise ValueError(f"metrics csv is empty: {metrics_path}")

    last = df.tail(1).iloc[0]
    final_loss = to_num(last.get("loss"))
    final_ppl = to_num(last.get("ppl"))

    # steps: prefer 'step' max; fallback to number of rows
    steps = None
    if "step" in df.columns:
        steps = to_num(df["step"].max())
    if steps is None:
        steps = int(len(df))

    # time_sec: sum 't' if exists
    time_sec = None
    if "t" in df.columns:
        s = pd.to_numeric(df["t"], errors="coerce").fillna(0).sum()
        time_sec = float(s)

    return dict(final_loss=final_loss, final_ppl=final_ppl, steps=steps, time_sec=time_sec)


def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Append a summarized row to results/ablations.csv")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--task", required=True, help="lm / mt / copy_task ...")
    parser.add_argument("--dataset", required=True, help="e.g., tiny_shakespeare")
    parser.add_argument("--tag", required=True, help="experiment tag, e.g., LM-fast / LM-noPE-120")
    parser.add_argument("--config", default=None, help="optional yaml to read model/train fields")

    # Optional CLI overrides for model fields
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--attn_type", type=str, default=None)
    parser.add_argument("--rel_pos", type=str, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--ablations", default=ABLATIONS_PATH, help="output csv (default results/ablations.csv)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_fields = collect_model_fields(cfg, args)
    tail = read_metrics_tail(args.metrics)

    row = {
        "task": args.task,
        "dataset": args.dataset,
        "tag": args.tag,

        # model/training description
        "d_model": model_fields.get("d_model", "NA"),
        "n_heads": model_fields.get("n_heads", "NA"),
        "n_layers": model_fields.get("n_layers", "NA"),
        "d_ff": model_fields.get("d_ff", "NA"),
        "max_len": model_fields.get("max_len", "NA"),
        "attn_type": model_fields.get("attn_type", "NA"),
        "rel_pos": model_fields.get("rel_pos", "NA"),
        "window": model_fields.get("window_size", "NA"),
        "max_steps": model_fields.get("max_steps", "NA"),
        "seed": model_fields.get("seed", "NA"),

        # results
        "final_loss": tail.get("final_loss", "NA"),
        "final_ppl": tail.get("final_ppl", "NA"),
        "time_sec": tail.get("time_sec", "NA"),
        "steps": tail.get("steps", "NA"),

        # traceability
        "metrics": args.metrics.replace("\\", "/"),
    }

    # CSV header order
    header = [
        "task", "dataset", "tag",
        "d_model", "n_heads", "n_layers", "d_ff", "max_len",
        "attn_type", "rel_pos", "window",
        "max_steps", "seed",
        "final_loss", "final_ppl", "time_sec", "steps",
        "metrics",
    ]

    ensure_parent(args.ablations)
    write_header = not os.path.isfile(args.ablations)

    # If exists and header differs, we keep our new header; old CSV will stay readable by pandas
    with open(args.ablations, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[OK] append -> {args.ablations}")
    print("Row:", row)


if __name__ == "__main__":
    main()
