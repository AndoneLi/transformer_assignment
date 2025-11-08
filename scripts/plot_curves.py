# scripts/plot_curves.py
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt

def read_series(p):
    xs, loss, ppl = [], [], []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(float(row["step"])))
            loss.append(float(row["loss"]))
            ppl.append(float(row["ppl"]))
    return xs, loss, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    xs, loss, ppl = read_series(args.metrics)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # loss
    plt.figure(); plt.plot(xs, loss)
    plt.xlabel("step"); plt.ylabel("loss"); plt.tight_layout()
    loss_png = out / ("mt_loss.png" if "mt" in str(out).lower() else "lm_loss.png")
    plt.savefig(loss_png); plt.close()

    # ppl
    plt.figure(); plt.plot(xs, ppl)
    plt.xlabel("step"); plt.ylabel("ppl"); plt.tight_layout()
    ppl_png = out / ("mt_ppl.png" if "mt" in str(out).lower() else "lm_ppl.png")
    plt.savefig(ppl_png); plt.close()

    print(f"[OK] saved: {loss_png} , {ppl_png}")

if __name__ == "__main__":
    main()
