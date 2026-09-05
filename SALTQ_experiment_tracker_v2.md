# SALT-Q Experiment Tracker v2

Llama-2-7B · 全部数值保留两位小数 · `—` = 未跑 / 不适用

---

## T1a  Commonsense-170k → 8 bench · INT3 g64 · bcal
Gap = (Avg − GPTQ floor) / (FP16 upper − GPTQ floor);Avg = MEAN(8) 未加权

| Method | Bits | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 upper(QLoRA merged) | 16 | 69.27 | 81.12 | 77.53 | 92.42 | 76.56 | 82.70 | 64.42 | 78.00 | **77.75** | 100% |
| **SALT-Q k=128**(salient_init=rtn) | 3.00 | 71.10 | 81.66 | 78.76 | 92.23 | 78.61 | 83.12 | 65.87 | 79.40 | **78.85** | **164%** |
| SALT-Q k=128(salient_init=gptq_latent) | 3.00 | 70.64 | 81.45 | 79.48 | 92.20 | 78.30 | 83.54 | 65.78 | 78.00 | 78.67 | 154% |
| QA-LoRA | 3.00 | 69.27 | 80.09 | 79.17 | 92.09 | 77.35 | 82.79 | 64.33 | 76.60 | 77.71 | 98% |
| QEFT k=128(lr 5e-5) | 3.35 | 69.51 | 80.20 | 77.33 | 91.87 | 77.11 | 82.20 | 65.61 | 77.60 | 77.68 | 96% |
| GPTQ floor(balanced 3500) | 3.00 | 67.25 | 79.11 | 76.56 | 91.32 | 74.66 | 80.51 | 63.82 | 75.20 | 76.05 | 0% |
| QEFT k=128(paper lr 5e-6) | 3.35 | 64.04 | 76.28 | 64.43 | 79.07 | 63.22 | 74.45 | 55.12 | 57.20 | 66.73 | — |
| LoTA-QAF(r=64, ω=48) | 3.00 | 62.08 | 71.71 | 57.88 | 35.41 | 49.72 | 70.08 | 48.89 | 52.80 | 56.07 | — |
| QEFT k=128 自身 floor(未微调) | 3.35 | 20.83 | 36.83 | 32.24 | 10.77 | 24.55 | 12.42 | 11.09 | 22.00 | 21.34 | — |
| LoTA-QAF 自身 floor(未微调) | 3.00 | 61.90 | 9.09 | 15.56 | 2.46 | 27.31 | 2.78 | 2.82 | 7.00 | 16.11 | — |

## T1b  Commonsense-170k → 8 bench · INT2 g32 · bcal

| Method | Bits | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 upper(QLoRA merged) | 16 | 69.33 | 80.74 | 77.53 | 92.62 | 76.09 | 82.07 | 65.44 | 78.20 | **77.75** | 100% |
| QEFT k=256(lr 5e-5 + 可训 zp) | 2.75 | 66.27 | 79.43 | 77.53 | 89.02 | 78.37 | 75.55 | 60.58 | 79.20 | **75.74** | **83%** |
| **SALT-Q k=256**(gptq_latent, zp×2) | 2.00 | 66.39 | 77.86 | 77.69 | 87.65 | 76.48 | 76.09 | 59.13 | 76.80 | **74.76** | **74%** |
| QEFT k=256(lr 5e-5) | 2.75 | 65.54 | 77.91 | 76.82 | 87.58 | 73.32 | 75.63 | 59.64 | 73.20 | 73.71 | 65% |
| QA-LoRA | 2.00 | 66.09 | 78.84 | 76.46 | 86.46 | 73.01 | 73.78 | 57.85 | 71.20 | 72.96 | 58% |
| GPTQ floor(balanced 3500) | 2.00 | 58.78 | 71.76 | 71.90 | 78.90 | 64.96 | 68.77 | 53.07 | 61.60 | 66.22 | 0% |
| QEFT k=256(paper lr 3.5e-6) | 2.75 | 62.11 | 69.21 | 58.50 | 46.50 | 53.04 | 65.78 | 44.54 | 50.40 | 56.26 | — |
| LoTA-QAF(r=64, ω=48) | 2.00 | 62.17 | 63.66 | 51.13 | 27.35 | 51.07 | 57.79 | 40.78 | 46.40 | 50.04 | — |
| QEFT k=256 自身 floor(未微调) | 2.75 | 1.19 | 3.65 | 20.73 | 8.26 | 20.68 | 6.73 | 6.40 | 13.20 | 10.10 | — |
| LoTA-QAF 自身 floor(未微调) | 2.00 | 5.38 | 13.38 | 15.76 | 5.77 | 8.84 | 4.29 | 3.33 | 13.00 | 8.72 | — |

---

## T2a  MetaMath → GSM8K / MATH · INT2 g32 · span bcal · 1 epoch
GSM8K n=1319,MATH n=5000,vLLM greedy;`—` 该臂未测该任务

| Method | Bits | GSM8K | MATH | train_loss |
|---|---|---|---|---|
| QLoRA fp16 upper(adapter 合并,未重量化) | 16 | **58.07** | 10.64 | 0.31 |
| **SALT-Q k=256**(gptq_latent, zp×2) | 2.00 | **56.48** | **13.68** | 0.24 |
| QA-LoRA | 2.00 | 52.54 | 11.04 | 0.26 |
| QEFT k=256(lr 5e-5,无 zp) | 2.75 | 48.52 | 7.44 | 0.30 |
| QLoRA→GPTQ floor(balanced 1k 校准) | 2.00 | 22.06 | 2.16 | — |
| QEFT k=256 自身 floor(未微调) | 2.75 | 0.00 | — | — |
| NF4 基座 + INT2 RTN(无 GPTQ,非本格行) | 2.00 | 0.00 | 0.00 | — |

## T2b  WikiText-2 PPL↓ · 1 epoch(主格)
在 train split 微调、报 test split PPL;非重叠窗口、全 token 计损

| Method | Bits | @2048 | @1024 | train_loss |
|---|---|---|---|---|
| QLoRA fp16 upper(adapter 合并,未重量化) | 16 | **4.84** | 5.08 | 1.67 |
| QEFT k=256 | 2.75 | **6.02** | 6.35 | 1.89 |
| QA-LoRA | 2.00 | **6.30** | 6.65 | 1.94 |
| **SALT-Q k=256** | 2.00 | **7.00** | 7.40 | 2.11 |
| QLoRA→GPTQ floor | 2.00 | **8.87** | 8.97 | — |
| NF4 基座 + INT2 RTN(无 GPTQ,非本格行) | 2.00 | 217446.74 | 217464.05 | — |

## T2c  WikiText-2 PPL↓ · 3 epoch

| Method | Bits | @2048 | @1024 | train_loss |
|---|---|---|---|---|
| QLoRA fp16 upper | 16 | **5.09** | 5.35 | 1.55 |
| QEFT k=256 | 2.75 | **6.68** | 7.08 | 1.58 |
| QA-LoRA | 2.00 | **8.82** | 9.45 | 1.45 |
| **SALT-Q k=256** | 2.00 | **10.15** | 10.89 | 1.24 |
| QLoRA→GPTQ floor | 2.00 | **10.27** | 11.01 | — |
| NF4 基座 + INT2 RTN(无 GPTQ,非本格行) | 2.00 | 206908.08 | 207016.52 | — |

## T2d  WikiText-2 PPL↓ · 未微调锚点

| 基座 | Bits | test @2048 | test @1024 | validation @2048 | validation @1024 |
|---|---|---|---|---|---|
| fp16 Llama-2-7B(协议锚点) | 16 | **5.47** | 6.10 | — | — |
| QEFT 基座(k=256 列留 fp16) | 2.75 | **7.54** | 8.51 | — | — |
| 纯 INT2 g32 GPTQ(0.5M in-domain 校准) | 2.00 | **13.44** | 14.04 | 13.74 | 13.89 |

## T2e  WikiText-2 过拟合诊断 · 3 epoch
train PPL = train split 前 166 窗 @2048(与 test 等代价)

| Method | 可训参数 | train PPL | validation | test | test/train |
|---|---|---|---|---|---|
| SALT-Q | 526.8M | 1.50 | 10.26 | 10.15 | 6.75 |
| QA-LoRA | 159.9M | 2.24 | 8.86 | 8.82 | 3.93 |
| QEFT | 348.1M | 3.52 | 6.74 | 6.68 | 1.90 |
| QLoRA fp16 upper | 159.9M(LoRA r=64) | 3.92 | 5.19 | 5.09 | 1.30 |
| GPTQ floor(未训) | 0 | 9.31 | 11.79 | 10.27 | 1.10 |
| 纯 INT2 裸基座(未训) | 0 | — | 13.74 | 13.44 | — |

## T2f  WikiText-2 过拟合诊断 · 1 epoch

| Method | 可训参数 | train PPL | validation | test | test/train |
|---|---|---|---|---|---|
| SALT-Q | 526.8M | 4.17 | 7.06 | 7.00 | 1.68 |
| QA-LoRA | 159.9M | 4.84 | 6.36 | 6.30 | 1.30 |
| QEFT | 348.1M | 4.81 | 6.08 | 6.02 | 1.25 |
| QLoRA fp16 upper | 159.9M(LoRA r=64) | 4.63 | 4.93 | 4.84 | 1.05 |
| GPTQ floor(未训) | 0 | 8.40 | 8.66 | 8.87 | 1.06 |

---

## T3a  训练成本 · Commonsense-170k · bcal · 1 epoch · 4×GPU · 147,580 records

| Method | Cell | train_runtime (s) | samples/s | train_loss | Peak Mem |
|---|---|---|---|---|---|
| QA-LoRA | INT3 g64 | 4286 | 34.43 | 1.23 | — |
| SALT-Q k=128(gptq_latent) | INT3 g64 | 5732 | 25.75 | 1.23 | — |
| SALT-Q k=128(rtn) | INT3 g64 | 5601 | 26.35 | 1.23 | — |
| QLoRA(upper/floor 母 ckpt) | INT3 g64 | 5840 | 25.27 | 1.24 | — |
| QEFT k=128(lr 5e-5) | INT3 g64 | 6802 | 21.70 | 1.24 | — |
| QEFT k=128(paper lr) | INT3 g64 | 6830 | 21.61 | 1.30 | — |
| LoTA-QAF | INT3 g64 | 17962 | 8.22 | 1.41 | — |
| QA-LoRA | INT2 g32 | 4400 | 33.54 | 1.27 | — |
| SALT-Q k=256(gptq_latent, zp×2) | INT2 g32 | 5920 | 24.93 | 1.26 | — |
| QLoRA(upper/floor 母 ckpt) | INT2 g32 | 5845 | 25.25 | 1.24 | — |
| QEFT k=256(lr 5e-5) | INT2 g32 | 6983 | 21.14 | 1.26 | — |
| QEFT k=256(paper lr) | INT2 g32 | 6989 | 21.12 | 1.34 | — |
| QEFT k=256(lr 5e-5 + 可训 zp) | INT2 g32 | 8921 | 16.54 | 1.25 | — |
| LoTA-QAF | INT2 g32 | 16565 | 8.91 | 1.67 | — |

## T3b  训练成本 · MetaMath / WikiText-2 · INT2 g32 · 4×GPU

| Method | MetaMath runtime (s) | MetaMath samples/s | Wiki2 1ep runtime (s) | Wiki2 1ep samples/s | Wiki2 3ep runtime (s) | Wiki2 3ep samples/s |
|---|---|---|---|---|---|---|
| QA-LoRA | 18610 | 21.22 | 229.3 | 12.24 | 682.7 | 12.34 |
| SALT-Q | 21600 | 18.29 | 275.6 | 10.19 | 821.8 | 10.25 |
| QLoRA(upper 母 ckpt) | 24030 | 16.44 | 300.2 | 9.35 | 888.0 | 9.48 |
| QEFT | 28800 | 13.71 | 205.1 | 13.68 | 610.7 | 13.79 |

## T3c  Pareto 汇总(CS-Avg 取 bcal 主表;GSM8K 取 MetaMath span-bcal 格)

| Method | Bits | g | CS-Avg | GSM8K | MATH | Wiki2 @2048 (1ep) | samples/s (CS) | W mat.? |
|---|---|---|---|---|---|---|---|---|
| QLoRA fp16 upper | 16 | — | 77.75 | 58.07 | 10.64 | 4.84 | 25.27 | No |
| QA-LoRA | 3 | 64 | 77.71 | — | — | — | 34.43 | No |
| QEFT | 3+fp16 | 64 | 77.68 | — | — | — | 21.70 | No |
| **SALT-Q** | 3 | 64 | **78.85** | — | — | — | 25.75 | No |
| LoTA-QAF | 3 | 64 | 56.07 | — | — | — | 8.22 | — |
| QA-LoRA | 2 | 32 | 72.96 | 52.54 | 11.04 | 6.30 | 33.54 | No |
| QEFT | 2+fp16 | 32 | 73.71 | 48.52 | 7.44 | 6.02 | 21.14 | No |
| QEFT + 可训 zp | 2+fp16 | 32 | 75.74 | — | — | — | 16.54 | No |
| **SALT-Q** | 2 | 32 | **74.76** | **56.48** | **13.68** | 7.00 | 24.93 | No |
| LoTA-QAF | 2 | 32 | 50.04 | — | — | — | 8.91 | — |
| LR-QAT | 3 | 64 | — | — | — | — | — | Yes |
| EfficientQAT | 3 | 64 | — | — | — | — | — | Yes(bw) |
| QWHA | 3 | 64 | — | — | — | — | — | — |

---

## T4a  SALT-Q 消融 · INT2 g32 · span bcal · k=256 · 1 epoch
基线 = 第一行;每行相对基线只改标注的一项

| Arm | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | train_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| salient_init=rtn(基线) | 65.05 | 76.55 | 75.64 | 81.77 | 71.27 | 69.99 | 53.33 | 69.00 | 70.32 | 1.30 |
| salient_init=gptq | 66.15 | 78.45 | 76.05 | 87.32 | 73.32 | 76.26 | 59.73 | 74.00 | 73.91 | 1.26 |
| salient_init=gptq, zp_lr ×2 | 66.64 | 78.62 | 77.33 | 87.39 | 75.14 | 76.52 | 59.13 | 74.60 | 74.42 | 1.25 |
| salient_init=gptq, zp_lr ×4 | 66.21 | 77.80 | 76.10 | 86.37 | 74.66 | 72.98 | 56.31 | 74.60 | 73.13 | 1.26 |
| salient_init=gptq, zp ×2 + salient_lr ×2 | 66.70 | 77.75 | 77.53 | 86.96 | 75.14 | 75.34 | 59.04 | 76.40 | 74.36 | 1.26 |
| **salient_init=gptq_latent, zp ×2** | 66.39 | 77.86 | 77.69 | 87.65 | 76.48 | 76.09 | 59.13 | 76.80 | **74.76** | 1.26 |
| salient_init=gptq_latent, zp 冻结(zp_lr=0) | 65.66 | 77.37 | 75.79 | 86.62 | 74.03 | 74.75 | 59.73 | 74.80 | 73.59 | 1.27 |

## T4b  SALT-Q 消融 · INT3 g64 · span bcal · k=128 · 1 epoch

| Arm | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | train_loss |
|---|---|---|---|---|---|---|---|---|---|---|
| **salient_init=rtn** | 71.10 | 81.66 | 78.76 | 92.23 | 78.61 | 83.12 | 65.87 | 79.40 | **78.85** | 1.23 |
| salient_init=gptq_latent | 70.64 | 81.45 | 79.48 | 92.20 | 78.30 | 83.54 | 65.78 | 78.00 | 78.67 | 1.23 |
| zp 冻结(zp_lr=0) | — | — | — | — | — | — | — | — | — | — |

## T4c  SALT-Q 早期消融 · INT3 g64 · response-only cell(非 bcal,只列 Avg)
基线 = `nogbl`(autoseg, k=128, salient_lr 3.46e-4, zp_lr 1.73e-3, group_by_length off)

| Arm | 改动 | Avg |
|---|---|---|
| 基线 nogbl | — | 79.04 |
| salient_lr 5e-5 | salient_lr | **81.62** |
| salient_lr 1e-4 | salient_lr | 81.46 |
| salient_lr 2e-4 | salient_lr | 80.60 |
| z-only | train_salient=false | 80.44 |
| zp_lr ×3 | zp_lr + scales_lr(两项) | 77.90 |
| legacy 分段 [2,30] | boundary_sizes | 78.87 |
| group_by_length=true | 采样器 | 36.97 |
| group_k=64 | 相对 salient_lr 5e-5 臂 | 81.33 |
| group_k=256 | 相对 salient_lr 5e-5 臂 | 81.98 |
| 3 epoch | epochs | 82.20 |
| QA-LoRA 对照 1ep / 3ep | — | 81.08 / 81.70 |

## T4d  SALT-Q 早期消融 · INT2 g32 · response-only / span cell(非 bcal,只列 Avg)

| Arm | Cell | Avg |
|---|---|---|
| 基线 ep1 | response-only | 64.68 |
| zp_lr ÷5 | response-only | 62.59 |
| zp_lr ×5 | response-only | 36.94 |
| zp 冻结 | response-only | 40.06 |
| z-only | response-only | 47.63 |
| zp ε-参数化 | response-only | 37.20 |
| zp-LoRA | response-only | 34.52 |
| zp-LoRA + 重校准 | response-only | 59.65 |
| salient-LoRA | response-only | 39.77 |
| 3 epoch | response-only | 72.24 |
| QA-LoRA 对照 | response-only | 71.26 |
| k=128 基线 | span | 67.16 |
| lr ↑ | span | 66.38 |
| lr ↓ | span | 65.75 |
| k=256 | span | 69.61 |
| k=128, 3 epoch | span | 70.61 |
| QA-LoRA 对照 | span | 67.11 |

---

## T5a  fp16-salient PTQ 扫描 · bcal · rank-ordered top-k(未微调显著列,PTQ only)
在同一 QLoRA-merged fp16 checkpoint 上导出;k=0 即 GPTQ floor

| k | fp16 share | INT3 g64 eff.bits | INT3 g64 Avg | INT2 g32 eff.bits | INT2 g32 Avg |
|---|---|---|---|---|---|
| 0(floor) | 0.00% | 3.00 | 76.05 | 2.00 | 66.22 |
| 32 | 0.61% | — | — | 2.09 | 69.68 |
| 64 | 1.21% | 3.16 | 76.42 | 2.17 | 69.72 |
| 128 | 2.43% | 3.32 | 76.32 | 2.34 | 70.39 |
| 256 | 4.86% | 3.63 | 75.98 | 2.68 | 69.99 |
| 512 | 9.72% | 4.26 | 76.35 | 3.36 | 70.65 |
| 1024 | 19.43% | 5.53 | 76.50 | 4.72 | 71.39 |
| 2048 | 38.86% | 8.05 | 76.96 | 7.44 | 72.57 |
| 全部 fp16 | 100.00% | 16 | 77.75 | 16 | 77.75 |

## T5b  解耦 2×2:salient 列的处理 × dense 部分的处理(bcal,Avg)
INT3 取 k=128,INT2 取 k=256(与各自主表同 k)

| Salient \ Dense | dense 静态 | dense z-trained |
|---|---|---|
| FP16 可训(QEFT 式) | INT3 77.68 / INT2 73.71 | INT3 — / INT2 75.74 |
| FP16 不训(PTQ 保护) | INT3 76.32 / INT2 69.99 | — |
| 同 bit QAT(ours) | INT3 — / INT2 73.59 | INT3 **78.85** / INT2 **74.76** |

## T6  salient 预算 k

| k | INT3 g64 bcal | INT2 g32 bcal | INT3 g64 resp-only | INT2 g32 span |
|---|---|---|---|---|
| 64 | — | — | 81.33 | — |
| 128 | **78.85** | — | 81.62 | 67.16 |
| 256 | — | **74.76** | 81.98 | 69.61 |
| 512 | — | — | — | — |
| random-128 对照 | — | — | — | — |

## T7  分段 / 置换消融

| 配置 | INT3 g64 resp-only | INT3 g64 bcal | INT2 g32 bcal | step time |
|---|---|---|---|---|
| auto segmentation(默认) | 79.04 | 78.85 | 74.76 | — |
| legacy 分段 [2,30] | 78.87 | — | — | — |
| per-layer P(oracle) | — | — | — | — |
| segment P ×2 / ×4 | — | — | — | — |
| 无 P(scatter-gather) | — | — | — | — |

---

## T8a  校准消融 · 同一 QLoRA-merged ckpt → GPTQ INT2 g32,只换校准集
探针 = 8 任务各 128 条的 answer loss / answer accuracy

| 校准集 | tokens | CS Avg | ans_loss | ans_acc |
|---|---|---|---|---|
| 旧 floor(BoolQ-only 128, first) | 9.5k | 36.64 | 1.17 | 0.41 |
| A BoolQ-128 masked | 9.5k | 35.12 | 1.76 | 0.38 |
| B BoolQ-only 3500 | 256k | 44.93 | 1.15 | 0.45 |
| C balanced 128 | 17k | 59.33 | 0.88 | 0.59 |
| **D balanced 3500** | 471k | **66.22** | 0.72 | 0.67 |
| E C4 128×2048(通用) | 262k | 30.67 | 1.10 | 0.45 |

## T8b  校准消融 · 未微调量化基座的 PPL(quantized base, no fine-tuning)

| 基座 | 校准集 | tokens | CS Avg | Wiki2 @1024 | Wiki2 @2048 | C4-val @2048 |
|---|---|---|---|---|---|---|
| fp16 参考(原始 Llama-2-7B) | — | — | — | 6.10 | **5.47** | 7.29 |
| fp16 母 ckpt(被量化前) | — | — | — | 6.56 | 5.82 | 7.62 |
| **D** INT2 g32 | balanced 3500 | 471k | **66.22** | 19.19 | **16.25** | **17.92** |
| **E** INT2 g32 | C4 128×2048 | 262k | **30.67** | 15.23 | **12.87** | **14.79** |
| **D** INT3 g64 | balanced 3500 | 471k | **76.05** | 7.85 | 6.90 | 8.85 |
| **E** INT3 g64 | C4 128×2048 | 262k | **72.07** | 7.61 | 6.68 | 8.64 |

---

## T9  重复评测(同一 checkpoint,两次独立评测)

| 对象 | 指标 | 第 1 次 | 第 2 次 | Δ |
|---|---|---|---|---|
| QEFT k=128 自身 floor(INT3 bcal) | CS Avg | 21.00 | 21.34 | 0.34 |
| QEFT k=256 自身 floor(INT2 bcal) | CS Avg | 10.09 | 10.10 | 0.01 |
| SALT-Q MetaMath(INT2 span bcal) | GSM8K | 56.33 | 56.48 | 0.15 |
| QEFT MetaMath(INT2 span bcal) | GSM8K | 48.90 | 48.52 | 0.38 |
| QEFT Wiki2 基座(未微调) | PPL @2048 | 7.54 | 7.54 | 0.00 |

---

## T10  未跑

| 项 | 状态 |
|---|---|
| QWHA(INT3 / INT2 复现) | 未跑 |
| EfficientQAT / LR-QAT | 未跑 |
| MetaMath INT3 g64 格 | 未跑 |
| Wiki2 INT3 g64 格 | 未跑 |
| 第二模型(Llama-3-8B / Qwen2.5-7B) | 未跑 |
| 3 seeds std | 未跑 |
| Peak memory 实测 / kernel microbench | 未跑 |
| T5b:FP16 不训 + dense z-trained | 未跑 |
| T6:k=512、random-k 对照 | 未跑 |
| T7:per-layer P、segment P ×2/×4、无 P | 未跑 |
