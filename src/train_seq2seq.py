# -*- coding: utf-8 -*-
"""
Seq2Seq training script (IWSLT17 En->De subset by default).
- YAML UTF-8 安全读取
- AdamW 超参显式转 float
- 对 get_mt_dataloaders / EncoderDecoder 做自适配（自动匹配参数名）
- AMP / grad clip / log & eval intervals
- 结果写入 results/mt/metrics.csv 与 results/ablations.csv
"""

import os, csv, math, time, argparse, random, inspect
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import yaml

from src.models.transformer import EncoderDecoder
from src.data import get_mt_dataloaders


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
    for k in candidates:
        if k in accepted:
            return k
    return None


# ---------- adapters ----------
def call_get_mt_dataloaders(args, tcfg, dcfg, seq_len, src_len, tgt_len):
    """根据 get_mt_dataloaders 的真实签名，只传它接受的参数名"""
    fn = get_mt_dataloaders
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())

    params = {}
    # name
    if "name" in accepted: params["name"] = args.dataset

    # batch size
    if "batch_size" in accepted: params["batch_size"] = int(tcfg.get("batch_size", 64))

    # seq/src/tgt len
    if "seq_len" in accepted:
        params["seq_len"] = int(seq_len)
    else:
        if "src_len" in accepted: params["src_len"] = int(src_len)
        if "tgt_len" in accepted: params["tgt_len"] = int(tgt_len)

    # languages（多种命名）
    lang_map = {
        "src_lang": args.source_lang, "source_lang": args.source_lang, "src_language": args.source_lang,
        "tgt_lang": args.target_lang, "target_lang": args.target_lang, "tgt_language": args.target_lang,
    }
    for k, v in lang_map.items():
        if k in accepted:
            params[k] = v

    # limits / workers
    if "limit_train" in accepted: params["limit_train"] = int(getattr(args, "limit_train", 0) or 0)
    if "limit_eval"  in accepted: params["limit_eval"]  = int(getattr(args, "limit_eval", 0) or 0)
    if "num_workers" in accepted: params["num_workers"] = 0

    print(f"[MT] get_mt_dataloaders kwargs -> {params}")
    return fn(**params)


def build_seq2seq_model(src_vocab, tgt_vocab, cfg, args, dev, seq_len):
    """对 EncoderDecoder 的构造函数做自适配"""
    m = cfg.get("model", {})
    d_model       = int(m.get("d_model", 512))
    n_heads       = int(m.get("n_heads", 8))
    n_layers      = int(m.get("n_layers", 6))
    n_enc_layers  = int(m.get("n_enc_layers", n_layers))
    n_dec_layers  = int(m.get("n_dec_layers", n_layers))
    d_ff          = int(m.get("d_ff", 2048))

    tcfg = cfg.get("train", {})
    attn_type     = (args.attn_type or tcfg.get("attn_type", "local")).lower()
    rel_pos       = (args.rel_pos or tcfg.get("rel_pos", "alibi")).lower()
    window_size   = int(args.window_size or tcfg.get("window_size", 64))
    dropout       = float(tcfg.get("dropout", 0.1))

    sig = inspect.signature(EncoderDecoder)
    accepted = set(sig.parameters.keys())

    kwargs = {}
    base = {
        "src_vocab_size": int(src_vocab),
        "tgt_vocab_size": int(tgt_vocab),
        "d_model": d_model,
        "n_heads": n_heads,
        "n_enc_layers": n_enc_layers,
        "n_dec_layers": n_dec_layers,
        "d_ff": d_ff,
        "dropout": dropout,
    }
    for k, v in base.items():
        if k in accepted: kwargs[k] = v

    # 可能需要长度键
    maxlen_key = _first_present(["max_len", "seq_len", "context_len", "max_seq_len"], accepted)
    if maxlen_key is not None:
        kwargs[maxlen_key] = int(seq_len)

    attn_key = _first_present(["attn_type","attention_type","attn"], accepted)
    if attn_key is not None: kwargs[attn_key] = attn_type

    relpos_key = _first_present(["rel_pos","relative_pos","relative_position","relpos"], accepted)
    if relpos_key is not None: kwargs[relpos_key] = rel_pos

    win_key = _first_present(["window_size","window","attn_window"], accepted)
    if win_key is not None: kwargs[win_key] = window_size

    print(f"[MT] EncoderDecoder kwargs -> { {k: kwargs[k] for k in kwargs if k in ['d_model','n_heads','n_enc_layers','n_dec_layers',maxlen_key,attn_key,relpos_key,win_key]} }")
    return EncoderDecoder(**kwargs).to(dev)


# ---------- training ----------
def train(cfg, args):
    set_seed(int(cfg["train"]["seed"]))
    dev = device_auto()
    precision = str(cfg["train"].get("precision", "amp")).lower()
    use_amp = (precision == "amp")

    tcfg = cfg.get("train", {}); dcfg = cfg.get("data", {})

    # 多源兜底取长度
    seq_len = (tcfg.get("seq_len") or dcfg.get("seq_len") or dcfg.get("max_seq_len") or dcfg.get("context_len") or 128)
    src_len = dcfg.get("src_len", seq_len)
    tgt_len = dcfg.get("tgt_len", seq_len)

    # dataloaders（自适配参数名）
    dl_train, dl_eval, src_vocab, tgt_vocab = call_get_mt_dataloaders(args, tcfg, dcfg, seq_len, src_len, tgt_len)

    # model（自适配构造）
    model = build_seq2seq_model(src_vocab, tgt_vocab, cfg, args, dev, seq_len)

    # loss / optimizer / scaler
    pad_id = int(dcfg.get("pad_id", 0))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    lr    = float(tcfg["lr"])
    betas = tuple(float(x) for x in tcfg["betas"])
    wd    = float(tcfg["weight_decay"])
    opt   = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    scaler = GradScaler(enabled=use_amp)

    # logging
    save_dir = tcfg.get("save_dir", "results/mt"); ensure_dir(save_dir)
    metrics_csv = Path(save_dir) / "metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "loss", "ppl", "lr", "time_sec"])

    log_interval  = int(tcfg.get("log_interval", 50))
    eval_interval = int(tcfg.get("eval_interval", 200))
    max_steps     = int(args.max_steps or tcfg.get("max_steps", 3000))
    warmup_steps  = int(tcfg.get("warmup_steps", 0))
    grad_clip     = float(tcfg.get("grad_clip", 1.0))

    # training loop
    model.train()
    t0 = time.time(); step = 0; running = 0.0
    train_iter = iter(dl_train)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train); batch = next(train_iter)

        if isinstance(batch, dict):
            src = batch.get("src").to(dev)
            tgt_in = batch.get("tgt_in").to(dev)
            tgt_out = batch.get("tgt_out").to(dev)
        else:
            src, tgt_in, tgt_out = [t.to(dev) for t in batch]

        # warmup
        if warmup_steps > 0 and step < warmup_steps:
            cur_lr = lr * (step + 1) / warmup_steps
        else:
            cur_lr = lr
        for pg in opt.param_groups: pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        step += 1; running += loss.item()

        if step % log_interval == 0 or step == 1:
            mean_loss = running / log_interval if step > 1 else loss.item()
            running = 0.0
            ppl = math.exp(min(20.0, mean_loss)); dt = time.time() - t0
            print(f"[MT] step {step:>6d}/{max_steps} | loss {mean_loss:.4f} | ppl {ppl:.2f} | lr {cur_lr:.2e} | t {dt:.1f}s")
            with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, f"{mean_loss:.6f}", f"{ppl:.6f}", f"{cur_lr:.6e}", f"{dt:.2f}"])

        if eval_interval > 0 and step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                try:
                    eb = next(iter(dl_eval))
                except StopIteration:
                    eb = next(iter(dl_eval))
                if isinstance(eb, dict):
                    es, ei, eo = eb["src"].to(dev), eb["tgt_in"].to(dev), eb["tgt_out"].to(dev)
                else:
                    es, ei, eo = [t.to(dev) for t in eb]
                logits = model(es, ei)
                l_eval = criterion(logits.reshape(-1, logits.size(-1)), eo.reshape(-1)).item()
                ppl_eval = math.exp(min(20.0, l_eval))
                print(f"[MT][eval] step {step} | loss {l_eval:.4f} | ppl {ppl_eval:.2f}")
            model.train()

    # final ablations row
    final_loss = loss.item(); final_ppl = math.exp(min(20.0, final_loss)); total_time = time.time() - t0
    ab_csv = Path("results") / "ablations.csv"; ensure_dir(ab_csv.parent)
    need_header = not ab_csv.exists()
    header = ["task","dataset","final_loss","final_ppl","time_sec"]
    row = ["mt", args.dataset, f"{final_loss:.6f}", f"{final_ppl:.6f}", f"{total_time:.2f}"]
    with open(ab_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        if need_header: w.writerow(header)
        w.writerow(row)

    print(f"[MT] done. metrics -> {metrics_csv} ; ablations -> {ab_csv}")


# ---------- cli ----------
def build_argparser():
    p = argparse.ArgumentParser(description="Train a Transformer Encoder-Decoder on IWSLT17.")
    p.add_argument("--config", type=str, default="configs/base_seq2seq.yaml")
    p.add_argument("--dataset", type=str, default="iwslt2017")
    p.add_argument("--source_lang", type=str, default="en")
    p.add_argument("--target_lang", type=str, default="de")
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_eval", type=int, default=0)
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
