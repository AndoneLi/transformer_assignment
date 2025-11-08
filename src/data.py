from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, text, seq_len=256):
        vocab = sorted(set(text))
        self.stoi = {ch:i+4 for i,ch in enumerate(vocab)}
        self.stoi['<pad>']=0; self.stoi['<bos>']=1; self.stoi['<eos>']=2; self.stoi['<unk>']=3
        ids = [self.stoi.get(ch,3) for ch in text]
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.vocab_size = len(self.stoi)
        self.seq_len = seq_len
    def __len__(self): return max(1, len(self.ids) - self.seq_len - 1)
    def __getitem__(self, idx):
        x = self.ids[idx:idx+self.seq_len]
        y = self.ids[idx+1:idx+self.seq_len+1]
        return x, y

def get_lm_dataloader(name="tiny_shakespeare", subset=None, seq_len=256, batch_size=32, num_workers=2, data_dir=None):
    if name == "tiny_shakespeare":
        # --- tiny_shakespeare 安全加载：远端优先，失败读本地 ---
        import os, io
        try:
            ds = load_dataset("tiny_shakespeare", cache_dir=data_dir)
            text = ds["train"]["text"][0]
        except Exception as e:
            print(f"[LM] remote tiny_shakespeare failed: {e}\n -> fallback to data/tiny_shakespeare.txt")
            local_fp = os.path.join("data", "tiny_shakespeare.txt")
            if not os.path.exists(local_fp):
                raise FileNotFoundError(f"未找到 {local_fp}；请联网或手动放入该文件后重试。")
            with io.open(local_fp, "r", encoding="utf-8") as f:
                text = f.read()
        # 下面保持原逻辑：构建 CharDataset / DataLoader
        dataset = CharDataset(text, seq_len=seq_len)
        vocab_size = dataset.vocab_size
    elif name == "wikitext":
        subset = subset or "wikitext-2-raw-v1"
        ds = load_dataset("wikitext", subset, cache_dir=data_dir)
        text = "\n".join(ds["train"]["text"])
        dataset = CharDataset(text, seq_len=seq_len)
        vocab_size = dataset.vocab_size
    else:
        raise ValueError("Unknown dataset")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader, vocab_size

def get_mt_dataloaders(name="iwslt2017", source_lang="en", target_lang="de", limit_train=10000, limit_eval=2000, batch_size=32, num_workers=2, data_dir=None, max_len=256):
    ds = load_dataset(name, cache_dir=data_dir, language_pair=(source_lang, target_lang))
    train = ds["train"].select(range(min(limit_train, len(ds["train"]))))
    valid = ds["validation"].select(range(min(limit_eval, len(ds["validation"]))))
    from transformers import AutoTokenizer
    tok_src = AutoTokenizer.from_pretrained("t5-small")
    tok_tgt = AutoTokenizer.from_pretrained("t5-small")
    tok_src.model_max_length = max_len; tok_tgt.model_max_length = max_len
    tok_src.pad_token = tok_src.eos_token; tok_tgt.pad_token = tok_tgt.eos_token
    def encode_split(split):
        inputs = tok_src(split["translation"][source_lang], truncation=True, padding="max_length", max_length=max_len)
        targets = tok_tgt(split["translation"][target_lang], truncation=True, padding="max_length", max_length=max_len)
        return {"src_ids": inputs["input_ids"], "tgt_ids": targets["input_ids"]}
    train = train.map(encode_split); valid = valid.map(encode_split)
    class MTDataset(Dataset):
        def __init__(self, hf_ds): self.ds = hf_ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            return torch.tensor(self.ds[i]["src_ids"]), torch.tensor(self.ds[i]["tgt_ids"])
    train_loader = DataLoader(MTDataset(train), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(MTDataset(valid), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, valid_loader, tok_src.vocab_size, tok_tgt.vocab_size
