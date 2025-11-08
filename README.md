Transformer 作业：语言模型（LM）与翻译模型（MT, Copy-Task）

本仓库提供一个可在 CPU 环境 下运行与复现的最小 Transformer 训练与可视化流程：

LM（语言模型）：在 Tiny Shakespeare 文本上训练与可视化；

MT（翻译模型）：本地 Copy-Task（零外网依赖）验证 Encoder–Decoder 管线；

统一 曲线导出（results/*/metrics.csv → results/*/*.png）与 汇总表（results/ablations.csv）便于验收。

1. 硬件与软件要求

最低硬件

CPU：任意近年的 x86_64 处理器（Intel/AMD）

内存：≥ 8 GB（推荐 16 GB）

磁盘：≥ 2 GB 可用空间

可选

NVIDIA GPU + CUDA（若无 GPU，代码会自动 fallback 到 CPU）

软件

操作系统：Windows 10/11，或 Linux/macOS

Python：3.9–3.11（建议与仓库 .venv/requirements.txt 一致）

PyTorch：CPU 版或 CUDA 版均可


2. 环境初始化
2.1 创建虚拟环境并安装依赖

Windows（PowerShell）

python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt


Linux/macOS（Bash/Zsh）

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


说明

运行脚本一律使用 模块模式（python -m xxx），避免 No module named 'src'。

3. 复现实验命令（Exact & Seed）

所有命令假定在项目根目录执行：llm_transformer_assignment_final/>

3.1 语言模型（LM，Tiny Shakespeare）

Windows

python -m src.train_lm --config configs\base_lm.yaml --dataset tiny_shakespeare --max_steps 300 --seed 42
python scripts\plot_curves.py --metrics results\lm\metrics.csv --outdir results\lm


Linux/macOS

python -m src.train_lm --config configs/base_lm.yaml --dataset tiny_shakespeare --max_steps 300 --seed 42
python scripts/plot_curves.py --metrics results/lm/metrics.csv --outdir results/lm


产物：

指标：results/lm/metrics.csv

曲线：results/lm/lm_loss.png、results/lm/lm_ppl.png

3.2 翻译模型（MT，本地 Copy-Task，零外网依赖）

Windows

python -m scripts.mt_copy_sanity --steps 400 --seed 7
python scripts\plot_curves.py --metrics results\mt\metrics.csv --outdir results\mt


Linux/macOS

python -m scripts.mt_copy_sanity --steps 400 --seed 7
python scripts/plot_curves.py --metrics results/mt/metrics.csv --outdir results/mt


产物：

指标：results/mt/metrics.csv

曲线：results/mt/mt_loss.png、results/mt/mt_ppl.png

3.3 汇总表（Ablations）

将 LM 与 MT(copy-task) 的末尾结果汇总到 results/ablations.csv。

Windows

python scripts\append_lm_row.py
python -c "import pandas as pd; print(pd.read_csv(r'results\ablations.csv').tail(5))"


Linux/macOS

python scripts/append_lm_row.py
python -c "import pandas as pd; print(pd.read_csv('results/ablations.csv').tail(5))"


预期 ablations.csv 至少包含两行：
mt, copy_task, … 与 lm, tiny_shakespeare, …

4. 自检

Windows

python -c "import pandas as pd, os; p1=r'results\lm\metrics.csv'; p2=r'results\mt\metrics.csv'; \
print(p1,'rows=',len(pd.read_csv(p1)),'png_loss=',os.path.exists(r'results\lm\lm_loss.png'),'png_ppl=',os.path.exists(r'results\lm\lm_ppl.png')); \
print(p2,'rows=',len(pd.read_csv(p2)),'png_loss=',os.path.exists(r'results\mt\mt_loss.png'),'png_ppl=',os.path.exists(r'results\mt\mt_ppl.png'))"


Linux/macOS

python - <<'PY'
import pandas as pd, os
for p in ['results/lm/metrics.csv','results/mt/metrics.csv']:
    print(p, 'rows=', len(pd.read_csv(p)),
          'png_loss=', os.path.exists(p.replace('metrics.csv','lm_loss.png') if 'lm' in p else 'results/mt/mt_loss.png'),
          'png_ppl=',  os.path.exists(p.replace('metrics.csv','lm_ppl.png')  if 'lm' in p else 'results/mt/mt_ppl.png'))
PY

5. 期望结果（参考）

LM（Tiny Shakespeare）：ppl ≈ 1.02（见 results/lm/*.png）

MT（Copy-Task）：steps=400 时 ppl ≈ 1.97（见 results/mt/*.png）

results/ablations.csv 至少两行（LM + MT）

注：LM 在 CPU 上时间相对长；MT Copy-Task 在 CPU 上也能很快收敛。若仅为验收，可将 --max_steps 调小以缩短时间。

6. 常见问题（FAQ）

No module named 'src'
用模块模式运行：python -m src.train_lm ...（不要直接 python src/train_lm.py）。

PowerShell 中多行 Python 报“不是内部或外部命令”
将多行保存为 .py 脚本再执行，或使用 Here-String：@" ... "@ | python -。

没有 CUDA
日志会提示禁用 AMP/GradScaler，程序自动在 CPU 上运行；不用改代码。

想进一步提速
可在 configs/base_lm.yaml 中把 d_model / n_layers / max_len / window_size 适当减小，或把 --max_steps 调小；不会影响“能跑通+出图+能汇总”的验收目标。

7. 目录结构（关键文件）
llm_transformer_assignment_final/
├─ configs/
│  └─ base_lm.yaml
├─ src/
│  └─ train_lm.py
├─ scripts/
│  ├─ mt_copy_sanity.py
│  ├─ plot_curves.py
│  └─ append_lm_row.py
├─ results/
│  ├─ lm/ (metrics.csv, lm_loss.png, lm_ppl.png)
│  ├─ mt/ (metrics.csv, mt_loss.png, mt_ppl.png)
│  └─ ablations.csv
└─ report/
   └─ main.tex
