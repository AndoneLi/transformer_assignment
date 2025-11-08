#!/usr/bin/env bash
set -e
# LM 主结果
python src/train_lm.py --config configs/base_lm.yaml --dataset tiny_shakespeare --max_steps 2000 --seed 42
# LM 消融：full
python src/train_lm.py --config configs/base_lm.yaml --dataset tiny_shakespeare --max_steps 1500 --seed 7 --attn_type full
# LM 消融：no relative pos
python src/train_lm.py --config configs/base_lm.yaml --dataset tiny_shakespeare --max_steps 1500 --seed 9 --rel_pos none
# MT 主结果
python src/train_seq2seq.py --config configs/base_seq2seq.yaml --dataset iwslt2017 --source_lang en --target_lang de --limit_train 10000 --limit_eval 2000 --max_steps 3000 --seed 7
# MT 窗口敏感性
python src/train_seq2seq.py --config configs/base_seq2seq.yaml --dataset iwslt2017 --source_lang en --target_lang de --limit_train 10000 --limit_eval 2000 --max_steps 2000 --seed 11 --window_size 32
