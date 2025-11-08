# scripts/add_tag.py
import argparse
import pandas as pd
from pathlib import Path

def infer_tag(row):
    attn = row.get("attn_type")  # e.g. local
    relp = row.get("rel_pos")    # e.g. alibi / sinusoid / none
    win  = row.get("window_size")  # e.g. 128 / 16
    ms   = row.get("max_steps")    # e.g. 180 / 120
    # 允许缺失
    attn = str(attn) if pd.notna(attn) else "attn?"
    relp = str(relp) if pd.notna(relp) else "pe?"
    win  = str(int(win)) if pd.notna(win) else "?"
    ms   = str(int(ms)) if pd.notna(ms) else "?"
    return f"{attn}-{relp}-w{win}-{ms}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="metrics.csv 路径（就地写回）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if "tag" not in df.columns:
        df["tag"] = df.apply(infer_tag, axis=1)
        df.to_csv(csv_path, index=False)
        print(f"[OK] added tag column -> {csv_path}")
    else:
        print("[OK] tag column already exists.")

if __name__ == "__main__":
    main()
