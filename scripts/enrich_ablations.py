# scripts/enrich_ablations.py
import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--abl", required=True, help="results/ablations.csv")
args = ap.parse_args()

p = Path(args.abl)
df = pd.read_csv(p)

if "tag" not in df.columns:
    def infer_tag(row):
        attn = row.get("attn_type")
        relp = row.get("rel_pos")
        win  = row.get("window_size")
        ms   = row.get("max_steps")
        attn = str(attn) if pd.notna(attn) else "attn?"
        relp = str(relp) if pd.notna(relp) else "pe?"
        win  = str(int(win)) if pd.notna(win) else "?"
        ms   = str(int(ms)) if pd.notna(ms) else "?"
        return f"{attn}-{relp}-w{win}-{ms}"
    df["tag"] = df.apply(infer_tag, axis=1)

# 仅对常见 tag 做映射
mapping = {
    "LM-global-120": {"attn_type": "global", "rel_pos": "alibi", "window_size": None, "max_steps": 120},
    "LM-noPE-120":   {"attn_type": "local",  "rel_pos": "none",  "window_size": 128,  "max_steps": 120},
    "LM-win16-120":  {"attn_type": "local",  "rel_pos": "alibi", "window_size": 16,   "max_steps": 120},
}

for tag, vals in mapping.items():
    m = (df["tag"] == tag)
    for k, v in vals.items():
        if k not in df.columns:
            df[k] = pd.NA
        df.loc[m & df[k].isna(), k] = v

# 写回
df.to_csv(p, index=False)
print(f"[OK] enriched -> {p}")
