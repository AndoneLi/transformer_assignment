# scripts/mt_copy_sanity.py
import math, time, random, argparse, csv, inspect
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from src.models.transformer import EncoderDecoder

def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def device_auto(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CopyDataset(Dataset):
    """合成平行语料：src == tgt（复制任务），固定长度填充。"""
    def __init__(self, n_samples=2000, vocab=200, seq_len=24, pad=0, bos=1):
        self.data = []
        for _ in range(n_samples):
            L = seq_len
            toks = [random.randint(2, vocab-1) for _ in range(L)]
            tgt = toks[:]                       # target = source
            tgt_in  = [bos] + tgt[:-1]          # teacher forcing 输入
            tgt_out = tgt
            self.data.append((
                torch.tensor(toks, dtype=torch.long),
                torch.tensor(tgt_in, dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long),
            ))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    src  = torch.stack([b[0] for b in batch], 0)
    tin  = torch.stack([b[1] for b in batch], 0)
    tout = torch.stack([b[2] for b in batch], 0)
    return {"src": src, "tgt_in": tin, "tgt_out": tout}

def _first(keys, accepted):
    for k in keys:
        if k in accepted: return k
    return None

def build_model(src_vocab, tgt_vocab, d_model=256, n_heads=4, n_layers=4, d_ff=1024,
                attn_type="local", rel_pos="alibi", window_size=32, seq_len=24, dropout=0.1, dev=None):
    import inspect
    sig = inspect.signature(EncoderDecoder)
    params = sig.parameters
    accepted = set(params.keys())

    # 组装可选 kwargs（只传模型真的接受的键）
    kw = {}
    base = {
        "d_model": d_model, "n_heads": n_heads,
        "n_enc_layers": n_layers, "n_dec_layers": n_layers,
        "d_ff": d_ff, "dropout": dropout,
    }
    for k, v in base.items():
        if k in accepted:
            kw[k] = v

    # seq/max_len 同义键
    for alias in ["max_len", "seq_len", "context_len", "max_seq_len"]:
        if alias in accepted:
            kw[alias] = int(seq_len)
            break

    # 注意力/相对位置/窗口 同义键
    for k_alias in (("attn_type","attention_type","attn"),
                    ("rel_pos","relative_pos","relative_position","relpos"),
                    ("window_size","window","attn_window")):
        for a in k_alias:
            if a in accepted:
                kw[a] = {"attn_type":attn_type, "relative_pos":rel_pos, "rel_pos":rel_pos,
                         "relpos":rel_pos, "window_size":window_size, "window":window_size,
                         "attn_window":window_size}.get(a, kw.get(a))
                break

    # 决定用“位置参数”还是“kwargs”传词表大小
    call_args = []
    if "src_vocab" in accepted and "tgt_vocab" in accepted:
        # 你的实现：要求位置/或显式关键词参数 src_vocab、tgt_vocab
        call_args = [int(src_vocab), int(tgt_vocab)]
    elif "src_vocab_size" in accepted and "tgt_vocab_size" in accepted:
        kw["src_vocab_size"] = int(src_vocab)
        kw["tgt_vocab_size"] = int(tgt_vocab)
    else:
        # 兜底：若只有一个 vocab_size（极少见）
        if "vocab_size" in accepted:
            kw["vocab_size"] = int(max(src_vocab, tgt_vocab))

    print("[MT-copy] model kwargs:", {k: kw[k] for k in kw})
    return EncoderDecoder(*call_args, **kw).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=24)
    ap.add_argument("--vocab", type=int, default=200)
    ap.add_argument("--save_dir", type=str, default="results/mt")
    args = ap.parse_args()

    set_seed(args.seed); dev = device_auto()
    train_ds = CopyDataset(n_samples=2000, vocab=args.vocab, seq_len=args.seq_len)
    eval_ds  = CopyDataset(n_samples=200,  vocab=args.vocab, seq_len=args.seq_len)
    dl_tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, collate_fn=collate)
    dl_ev = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = build_model(args.vocab, args.vocab, seq_len=args.seq_len, dev=dev)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    opt = AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.01)
    scaler = GradScaler(enabled=(torch.cuda.is_available()))
    ensure_dir(args.save_dir)
    mfile = Path(args.save_dir)/"metrics.csv"
    if not mfile.exists():
        with open(mfile,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["step","loss","ppl","lr","time_sec"])

    t0=time.time(); step=0; it=iter(dl_tr); running=0.0; log_int=50
    model.train()
    while step < args.steps:
        try: batch = next(it)
        except StopIteration:
            it = iter(dl_tr); batch = next(it)
        src, tin, tout = batch["src"].to(dev), batch["tgt_in"].to(dev), batch["tgt_out"].to(dev)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=(torch.cuda.is_available())):
            logits = model(src, tin)                # [B, L, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tout.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        step += 1; running += loss.item()

        if step % log_int == 0 or step == 1:
            mean = running / log_int if step>1 else loss.item(); running=0.0
            ppl = math.exp(min(20.0, mean)); dt = time.time()-t0
            print(f"[MT-copy] step {step}/{args.steps} | loss {mean:.4f} | ppl {ppl:.2f} | t {dt:.1f}s")
            with open(mfile,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([step, f"{mean:.6f}", f"{ppl:.6f}", f"{3e-4:.6e}", f"{dt:.2f}"])

    # 写 ablations.csv
    ab = Path("results")/"ablations.csv"; ab.parent.mkdir(parents=True, exist_ok=True)
    need_header = not ab.exists()
    with open(mfile,"r",encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    with open(ab,"a",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["task","dataset","d_model","n_heads","n_layers","attn_type","window","rel_pos","final_loss","final_ppl","time_sec"])
        w.writerow(["mt","copy_task","-","-","-","local","32","alibi", last["loss"], last["ppl"], last["time_sec"]])
    print("[MT-copy] done. metrics ->", mfile, "; appended a row to ->", ab)

if __name__ == "__main__":
    main()
