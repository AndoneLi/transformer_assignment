# scripts/make_ablation_table.py
import argparse, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--abl", required=True, help="results/ablations.csv")
ap.add_argument("--tex", required=True, help="输出的 LaTeX 文件路径")
args = ap.parse_args()

df = pd.read_csv(args.abl)

# 如果 ablations.csv 没有 tag，就按常用字段现场推断一个
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

# 只取会用到的列（存在则取）
cols = ["tag", "attn_type", "rel_pos", "window_size", "final_loss", "final_ppl", "time_sec", "dataset", "task"]
cols = [c for c in cols if c in df.columns]
out = df[cols].copy()

tex = Path(args.tex)
with open(tex, "w", encoding="utf-8") as f:
    f.write(out.to_latex(index=False, float_format="%.4f"))
print(f"[OK] wrote -> {tex}")
