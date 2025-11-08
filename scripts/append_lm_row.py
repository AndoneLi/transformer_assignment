# scripts/append_lm_row.py
import os, csv, pandas as pd

ab = r"results/ablations.csv"
m  = r"results/lm/metrics.csv"

# 读取 LM 的最后一个指标点
last = pd.read_csv(m).tail(1).iloc[0]
need_header = not os.path.exists(ab)

with open(ab, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if need_header:
        w.writerow(["task","dataset","d_model","n_heads","n_layers","attn_type","window","rel_pos","final_loss","final_ppl","time_sec"])
    # 结构型字段先用 '-' 占位即可
    w.writerow(["lm","tiny_shakespeare","-","-","-","-","-","-",
                f"{float(last['loss']):.6f}", f"{float(last['ppl']):.6f}", f"{float(last['time_sec']):.2f}"])
print("OK: appended LM row ->", ab)
