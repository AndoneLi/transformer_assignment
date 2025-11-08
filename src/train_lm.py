import torch, os
torch.set_num_threads(max(1, os.cpu_count() or 4))
torch.set_num_interop_threads(max(1, (os.cpu_count() or 4)//2))
# -*- coding: utf-8 -*-
"""
Language Modeling training script (tiny_shakespeare by default).
- YAML UTF-8 安全读取
- AdamW 超参显式转 float
- 对 EncoderOnlyLM 的 __init__ 做自适配（自动匹配 max_len/seq_len/context_len 等同义键）
- AMP / grad clip / log & eval intervals
- 结果写入 results/lm/metrics.csv 与 results/ablations.csv

Run:
  python -m src.train_lm --config configs/base_lm.yaml --dataset tiny_shakespeare --max_steps 2000 --seed 42
"""

import os, csv, math, time, argparse, random, inspect
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import yaml

from src.models.transformer import EncoderOnlyLM
from src.data import get_lm_dataloader


# ---------- utils ----------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_yaml_utf8(path: str):
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _first_present(candidates, accepted):
    """从 candidates 中挑第一个被目标函数接受的参数名"""
    for k in candidates:
        if k in accepted:
            return k
    return None

def build_lm_model(vocab_size, cfg, args, seq_len, dev):
    """对 EncoderOnlyLM 的构造函数做自适配，只传它真的接受的参数名"""
    m = cfg.get("model", {})
    d_model     = int(m.get("d_model", 512))
    n_layers    = int(m.get("n_layers", 8))
    n_heads     = int(m.get("n_heads", 8))
    d_ff        = int(m.get("d_ff", 2048))
    dropout     = float(cfg["train"].get("dropout", 0.1))
    attn_type   = (args.attn_type or cfg["train"].get("attn_type", "local")).lower()
    rel_pos     = (args.rel_pos or cfg["train"].get("rel_pos", "alibi")).lower()
    window_size = int(args.window_size or cfg["train"].get("window_size", 128))

    sig = inspect.signature(EncoderOnlyLM)
    accepted = set(sig.parameters.keys())

    # 基础必备键
    kwargs = {}
    for k, v in {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "dropout": dropout,
    }.items():
        if k in accepted:
            kwargs[k] = v

    # max_len / seq_len / context_len / max_seq_len 同义键
    maxlen_key = _first_present(["max_len", "seq_len", "context_len", "max_seq_len"], accepted)
    if maxlen_key is not None:
        kwargs[maxlen_key] = int(seq_len)

    # 注意力类型/窗口/相对位置的同义键
    attn_key   = _first_present(["attn_type", "attention_type", "attn"], accepted)
    if attn_key is not None: kwargs[attn_key] = attn_type

    relpos_key = _first_present(["rel_pos", "relative_pos", "relative_position", "relpos"], accepted)
    if relpos_key is not None: kwargs[relpos_key] = rel_pos

    win_key    = _first_present(["window_size", "window", "attn_window"], accepted)
    if win_key is not None: kwargs[win_key] = window_size

    model = EncoderOnlyLM(**kwargs).to(dev)
    # 便于排查：打印一次映射后的关键超参
    print(f"[LM] model kwargs -> { {k: kwargs[k] for k in kwargs if k in ['d_model','n_layers','n_heads','d_ff',maxlen_key,attn_key,relpos_key,win_key] } }")
    return model


# ---------- training ----------
def train(cfg, args):
    set_seed(int(cfg["train"]["seed"]))
    dev = device_auto()
    precision = str(cfg["train"].get("precision", "amp")).lower()
    use_amp = (precision == "amp")

    # dataloader
    dataset_name = args.dataset
    batch_size = int(cfg["train"]["batch_size"])
    seq_len = int(cfg["train"].get("seq_len", 256))
    loader, vocab_size = get_lm_dataloader(name=dataset_name, batch_size=batch_size, seq_len=seq_len, num_workers=0)

    # model（自适配构造）
    model = build_lm_model(vocab_size, cfg, args, seq_len, dev)

    # loss / optimizer / scaler
    criterion = nn.CrossEntropyLoss()
    lr    = float(cfg["train"]["lr"])
    betas = tuple(float(x) for x in cfg["train"]["betas"])
    wd    = float(cfg["train"]["weight_decay"])
    opt   = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scaler = GradScaler(enabled=use_amp)

    # logging & save
    save_dir = cfg["train"].get("save_dir", "results/lm"); ensure_dir(save_dir)
    metrics_csv = Path(save_dir) / "metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "loss", "ppl", "lr", "time_sec"])

    log_interval  = int(cfg["train"].get("log_interval", 50))
    eval_interval = int(cfg["train"].get("eval_interval", 200))
    max_steps     = int(args.max_steps or cfg["train"]["max_steps"])
    warmup_steps  = int(cfg["train"].get("warmup_steps", 0))
    grad_clip     = float(cfg["train"].get("grad_clip", 1.0))

    model.train()
    global_step = 0; t0 = time.time(); running_loss = 0.0

    while global_step < max_steps:
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)

            # warmup
            if warmup_steps > 0 and global_step < warmup_steps:
                cur_lr = lr * (global_step + 1) / warmup_steps
            else:
                cur_lr = lr
            for pg in opt.param_groups: pg["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with autocast():
                    logits = model(xb)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update()
            else:
                logits = model(xb); loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            running_loss += loss.item(); global_step += 1

            if global_step % log_interval == 0 or global_step == 1:
                mean_loss = running_loss / log_interval if global_step > 1 else loss.item()
                running_loss = 0.0
                ppl = math.exp(min(20.0, mean_loss)); dt = time.time() - t0
                print(f"[LM] step {global_step:>6d}/{max_steps} | loss {mean_loss:.4f} | ppl {ppl:.2f} | lr {cur_lr:.2e} | t {dt:.1f}s")
                with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([global_step, f"{mean_loss:.6f}", f"{ppl:.6f}", f"{cur_lr:.6e}", f"{dt:.2f}"])

            if eval_interval > 0 and global_step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    logits = model(xb)
                    l_eval = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)).item()
                    ppl_eval = math.exp(min(20.0, l_eval))
                    print(f"[LM][eval] step {global_step} | loss {l_eval:.4f} | ppl {ppl_eval:.2f}")
                model.train()

            if global_step >= max_steps: break

    final_loss = loss.item(); final_ppl = math.exp(min(20.0, final_loss)); total_time = time.time() - t0

    ab_csv = Path("results") / "ablations.csv"; ensure_dir(ab_csv.parent)
    need_header = not ab_csv.exists()
    header = ["task","dataset","final_loss","final_ppl","time_sec"]
    row = ["lm", dataset_name, f"{final_loss:.6f}", f"{final_ppl:.6f}", f"{total_time:.2f}"]
    with open(ab_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        if need_header: w.writerow(header)
        w.writerow(row)
    print(f"[LM] done. metrics -> {metrics_csv} ; ablations -> {ab_csv}")


# ---------- cli ----------
def build_argparser():
    p = argparse.ArgumentParser(description="Train a character-level LM on tiny_shakespeare.")
    p.add_argument("--config", type=str, default="configs/base_lm.yaml")
    p.add_argument("--dataset", type=str, default="tiny_shakespeare")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--attn_type", type=str, default=None, help="local|full")
    p.add_argument("--rel_pos", type=str, default=None, help="alibi|none")
    p.add_argument("--window_size", type=int, default=None)
    return p

def main():
    args = build_argparser().parse_args()
    cfg = load_yaml_utf8(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    train(cfg, args)

if __name__ == "__main__":
    main()
