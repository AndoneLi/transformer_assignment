import math, random, os, time
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(name: str = "auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def warmup_cosine_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def log_metrics(save_dir, metrics):
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "metrics.csv"
    df = pd.DataFrame([metrics])
    if not path.exists(): df.to_csv(path, index=False)
    else: df.to_csv(path, index=False, mode='a', header=False)

def log_ablation_row(row: dict, out_csv: str = 'results/ablations.csv'):
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if not out.exists(): df.to_csv(out, index=False)
    else: df.to_csv(out, index=False, mode='a', header=False)

def save_curve(csv_path: str, png_path: str, x_col: str, y_cols):
    df = pd.read_csv(csv_path)
    plt.figure()
    for y in y_cols:
        if y in df.columns: plt.plot(df[x_col], df[y], label=y)
    plt.xlabel(x_col); plt.ylabel(",".join(y_cols)); plt.legend(); plt.tight_layout()
    plt.savefig(png_path, dpi=180)
