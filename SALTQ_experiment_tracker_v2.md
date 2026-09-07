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

## T2a2  MetaMath → GSM8K / MATH · INT3 g64 · span bcal · 1 epoch
本地 2 卡(4×5×4 → 8×5×2,eff batch 仍是 80,T=4938)。vLLM greedy,GSM8K n=1319、MATH n=5000。

| Method | Bits | g | k | zp_lr | GSM8K | MATH |
|---|---|---|---|---|---|---|
| QLoRA fp16 upper(同一份母 ckpt,与 INT2 格共用) | 16 | — | — | — | 58.07 | 10.64 |
| **SALT-Q(zp×2,主行)** | 3 | 64 | 128 | 3.46e-3(**2×**) | **58.30** | **12.56** |
| SALT-Q(zp 1×,已作废为主行) | 3 | 64 | 128 | 1.73e-3(1×) | 56.18 | 10.90 |
| QA-LoRA 对照 | 3 | 64 | — | — | 待跑 | 待跑 |
| *对照:SALT-Q INT2 g32(T2a)* | *2* | *32* | *256* | *3.46e-3(**2×**)* | *56.48* | *13.68* |

**INT3 反而不如自己的 INT2 兄弟(MATH 10.90 vs 13.68,约 6 个标准误;GSM8K −0.30 在噪声内)。**
这是倒过来的,但**不能归因于比特宽度**——两格差了四件事,而且三件都对 INT3 不利:

| | INT2 格 | INT3 格 | 对 INT3 的影响 |
|---|---|---|---|
| zp_lr | 3.46e-3(2×) | 1.73e-3(1×) | z 档欠驱动,见下 |
| group_k | 256 | 128 | salient 可训权重 314.6M → **157.3M** |
| group_size | 32 | 64 | 可训 z 约 202M → **101.2M** |
| salient_lr | 1.25e-4 | 5.0e-5 | 按 1/(2^b−1) 律,设计如此 |

**zp 欠驱动是实测的,不是推断。** 用 `scripts/measure_saltq_displacement.py` 量已完成的 INT3
checkpoint(零 GPU 训练成本):

| 档 | 实测 p50 | 目标 | 判定 |
|---|---|---|---|
| `\|Δz_N\|` | **0.058 levels** | 0.1–0.3 | **欠驱动 2–5×** |
| `\|ΔW_S\|` | 0.134 grid steps | ~0.5(MetaMath 口径) | 偏低,但目标本身跨数据集不成立(见下) |
| `\|Δs_S\|` | 4.21 % | ~3 % | 在靶心 |

zp_lr 3.46e-3 × 实测 c(T=4938 下约 33)≈ 0.115 levels,正好是 INT2 兄弟拿到的驱动强度。

**已跑,预测-干预-验证闭环成立。** `configs/saltq_mm_int3_g64_ep1_span_bcal_sgptql_zp2x.yaml`
只改 `zp_lr_by_bits[3]` 一个键(两个离线基座与父 run 逐位共用,所以确实是单变量):

| | 父 run(zp 1×) | zp×2 | Δ |
|---|---|---|---|
| 实测 `\|Δz_N\|` p50 | 0.058 levels(欠驱动) | **0.101 levels(进入 0.1–0.3 靶区)** | 预测过 ~0.115 |
| 实测 `\|ΔW_S\|` p50 | 0.134 grid steps | 0.121(未动 salient_lr,如设计) | — |
| 实测 `\|Δs_S\|` p50 | 4.21 % | 3.94 %(靶心) | — |
| GSM8K | 56.18 | **58.30** | **+2.12** |
| MATH | 10.90 | **12.56** | **+1.66** |

**而且 zp×2 在两个指标上都越过了 fp16 上界**(58.30 > 58.07,12.56 > 10.64)。对 INT2 兄弟:
GSM8K 反超(58.30 > 56.48),MATH 仍低(12.56 < 13.68)——注意两格仍差 group_k / group_size
两项,不是纯比特对照。

**方法论价值高于这个数字**:AGENTS.md 那句「先用 `measure_saltq_displacement.py` 量位移,
再决定动哪个 lr」在这里又兑现了一次——零 GPU 成本诊断出唯一欠驱动的一档,单键修正,+2.12。

> `|ΔW_S|` 这一档**先不动**:0.5 的目标是 MetaMath 推出来的,而这套配方的 salient_lr 5e-5 是从
> Commonsense 搬来的,在那边同一把尺子量到的最好成绩(81.46)恰恰出现在 **0.079 steps**,而
> 落在推荐区间里的 0.273 反而最差(79.04)。目标随数据集变,未定,动它会让 run 变成双变量。

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

## T2b2  WikiText-2 PPL↓ · INT3 g64 · 1 epoch
与 T2b 同协议(train split 微调、报 test split PPL、非重叠窗口、全 token 计损),唯一变化是量化格:
bits 2→3、group_size 32→64、salient/weak 预算 256→128。effective batch 仍是 16(本地单卡
4×20×1),T=176,所有 lr 原样搬运。

| Method | Bits | @2048 | @1024 | 对 floor |
|---|---|---|---|---|
| QLoRA fp16 upper(adapter 合并,未重量化) | 16 | **4.84** | 5.07 | −0.43 |
| QEFT k=128 | 3.35 | **5.11** | 5.36 | −0.16 |
| QA-LoRA | 3 | **5.13** | 5.39 | −0.14 |
| **SALT-Q k=128** | 3 | **5.16** | 5.41 | **−0.11** |
| QLoRA→GPTQ floor(in-domain 256×2048 校准) | 3 | **5.27** | 5.54 | 0 |
| QEFT 基座(k=128 列留 fp16,未微调) | 3.35 | 5.79 | 6.46 | +0.52 |
| NF4 基座 + INT3 RTN(无 GPTQ,非本格行) | 3 | 99518.29 | 67346.40 | — |

**这一格只有 0.43 宽**(INT2 那一格是 4.03),三个微调臂全部落在彼此 0.05 以内。SALT-Q 高出
floor 0.11,但落后 QA-LoRA 0.03、落后 QEFT 0.05——在这个宽度下三者不可区分。这与 T2e/T2f 的
过拟合诊断一致:Wiki2 训练集只有 2.87M token,SALT-Q 的 527M 可训参数在这一列本来就是劣势,
多给一个比特只是把可争的空间进一步压小。**不要把这一格读成"SALT-Q 在 3bit 语言建模上更好"。**

> fp16 upper 的 4.84 / 5.07 与 T2b 的 INT2 格逐位一致(4.84 / 5.08)——它本来就与比特无关,
> 这是本地单卡 batch 重排(4×1×4 → 4×20×1)没有扰动训练的独立佐证。

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

## T3a  训练成本 · Commonsense-170k · bcal · 1 epoch · 147,580 records

> **⚠ 这张表原来的 `samples/s` 一列不可直接比较,标题里的「4×GPU」是错的。** 三种方法跑在
> **三种卡数**上:QLoRA / QA-LoRA / SALT-Q 用 4 卡,**QEFT 全部用 2 卡**
> (`jobs/cs_qeft_*.pbs` 里写死 `--num_gpus 2`,eff batch 靠 `5 × 8 × 2 = 80` 补回来),
> **LoTA-QAF 用 1 卡**(`ngpus=1`)。原表把三者的原始吞吐并排,于是同时得出了两个假结论:
> 「QEFT 比 SALT-Q 慢」和「LoTA-QAF 慢 4 倍」。两个都是卡数假象。
>
> 下面加了 `GPU` 与 `samples/s/GPU` 两列。**按卡归一后 QEFT 是最快的一档,SALT-Q 最慢。**
> 归一本身也不精确(DDP 扩展是次线性的,卡越多每卡吞吐越低),所以它**低估**了 4 卡那几行;
> 真实位置在原始值与归一值之间。但方向有独立佐证:T3b 的 WikiText-2 列**四个臂全部是 4 卡**,
> 排序是 QEFT 13.68 > QA-LoRA 12.24 > SALT-Q 10.19 > QLoRA 9.35 —— 与这里归一后的排序
> (QEFT 10.6–10.9 > QA-LoRA 8.4–8.6 > SALT-Q 6.2–6.6 ≈ QLoRA 6.3)完全一致。

| Method | Cell | GPU | train_runtime (s) | samples/s | **samples/s/GPU** | train_loss | Peak Mem |
|---|---|---|---|---|---|---|---|
| QA-LoRA | INT3 g64 | 4 | 4286 | 34.43 | 8.61 | 1.23 | — |
| SALT-Q k=128(gptq_latent) | INT3 g64 | 4 | 5732 | 25.75 | **6.44** | 1.23 | — |
| SALT-Q k=128(rtn) | INT3 g64 | 4 | 5601 | 26.35 | 6.59 | 1.23 | — |
| QLoRA(upper/floor 母 ckpt) | INT3 g64 | 4 | 5840 | 25.27 | 6.32 | 1.24 | — |
| QEFT k=128(lr 5e-5) | INT3 g64 | **2** | 6802 | 21.70 | **10.85** | 1.24 | — |
| QEFT k=128(paper lr) | INT3 g64 | **2** | 6830 | 21.61 | **10.81** | 1.30 | — |
| LoTA-QAF | INT3 g64 | **1** | 17962 | 8.22 | **8.22** | 1.41 | — |
| QA-LoRA | INT2 g32 | 4 | 4400 | 33.54 | 8.39 | 1.27 | — |
| SALT-Q k=256(gptq_latent, zp×2) | INT2 g32 | 4 | 5920 | 24.93 | **6.23** | 1.26 | — |
| QLoRA(upper/floor 母 ckpt) | INT2 g32 | 4 | 5845 | 25.25 | 6.31 | 1.24 | — |
| QEFT k=256(lr 5e-5) | INT2 g32 | **2** | 6983 | 21.14 | **10.57** | 1.26 | — |
| QEFT k=256(paper lr) | INT2 g32 | **2** | 6989 | 21.12 | **10.56** | 1.34 | — |
| QEFT k=256(lr 5e-5 + 可训 zp) | INT2 g32 | **2** | 8921 | 16.54 | **8.27** | 1.25 | — |
| LoTA-QAF | INT2 g32 | **1** | 16565 | 8.91 | **8.91** | 1.67 | — |

## T3b  训练成本 · MetaMath / WikiText-2 · INT2 g32
> MetaMath 那两列与 T3a 有同一个坑:**QEFT 用 2 卡**(`jobs/cs_qeft_mm_int2_lr5e5.pbs`),其余 4 卡。
> 按卡归一:QEFT 6.86 > QA-LoRA 5.31 > SALT-Q 4.57 > QLoRA 4.11 samples/s/GPU。
> **WikiText-2 两列是干净的——四个臂全是 4 卡**,可以直接读,而它给出的排序与上面的归一一致。
> 这一列因此是本仓库里唯一一处未被卡数污染的方法间速度比较。

| Method | MetaMath runtime (s) | MetaMath samples/s | Wiki2 1ep runtime (s) | Wiki2 1ep samples/s | Wiki2 3ep runtime (s) | Wiki2 3ep samples/s |
|---|---|---|---|---|---|---|
| QA-LoRA | 18610 | 21.22 | 229.3 | 12.24 | 682.7 | 12.34 |
| SALT-Q | 21600 | 18.29 | 275.6 | 10.19 | 821.8 | 10.25 |
| QLoRA(upper 母 ckpt) | 24030 | 16.44 | 300.2 | 9.35 | 888.0 | 9.48 |
| QEFT | 28800 | 13.71 | 205.1 | 13.68 | 610.7 | 13.79 |

## T3c  Pareto 汇总(CS-Avg 取 bcal 主表;GSM8K 取 MetaMath span-bcal 格)

> `samples/s (CS)` 沿用 T3a 的原始值,**跨方法不可比**(QEFT 2 卡 / LoTA-QAF 1 卡 / 其余 4 卡)。
> 要比速度请读 T3a 的 `samples/s/GPU` 或 T2/T3b 的 WikiText-2 列。

| Method | Bits | g | CS-Avg | GSM8K | MATH | Wiki2 @2048 (1ep) | samples/s (CS, 卡数不齐) | W mat.? |
|---|---|---|---|---|---|---|---|---|
| QLoRA fp16 upper | 16 | — | 77.75 | 58.07 | 10.64 | 4.84 | 25.27 | No |
| QA-LoRA | 3 | 64 | 77.71 | — | — | 5.13 | 34.43 | No |
| QEFT | 3+fp16 | 64 | 77.68 | — | — | 5.11 | 21.70 | No |
| **SALT-Q** | 3 | 64 | **78.85** | — | — | 5.16 | 25.75 | No |
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
消融的自变量是**残差置换的个数**。P_k 离线折进权重,只有 segment 边界折不掉,需要运行时
`index_select`(BoundaryGatherHook),所以 N 段 = 每次 forward N−1 次 gather——这就是 step time 那一列。
用 `qat.saltq.force_segments` 强制段数:DP、目标函数、sigma-outlier 选择器全部不变,只固定预算,
返回的仍是该预算下的最优切分(`force_segments` = DP 自己的选择时逐位复现默认切分)。

**默认那一行就是 ×2。** group_k=256 下 DP 只买了两段(`boundary_sizes=[2, 30]`):overflow 在第二段
之后不再下降,而 tie-break 取达到最小值的最小段数。所以 ×2 无需再跑,而 ×4 是 DP 主动放弃的那个点,
不是默认的换名。

| 配置 | 段数 | runtime gather | INT3 g64 resp-only | INT3 g64 bcal | INT2 g32 bcal | step time |
|---|---|---|---|---|---|---|
| segment P ×1(单一全局 P) | 1 | **0** | — | — | **74.43** | 24900 s(5.927 samples/s) |
| **auto segmentation(默认)= ×2** | 2 | 1 | 79.04 | 78.85 | **74.76** | — |
| segment P ×4 | 4 | 3 | — | — | 未跑(见下) | — |
| *(INT3 列的 ×1 / ×32,见下表)* | | | | | | |
| per-layer P(**32 个 P**,attn/mlp 共用) | 32 | 31 | — | — | **75.02** | 25190 s(5.859 samples/s) |
| **真·per-site P(64 个 P)** | 64 | 63 | — | — | **未实现**(见下) | — |
| legacy 分段 [2,30] | 2 | 1 | 78.87 | — | — | — |
| 无 P(scatter-gather) | — | — | — | — | — | — |

### INT3 g64 bcal 列(transfer test:预算绷紧的那一格)
INT2 那一格 k=256 宽松(DP 只买 2 段、overflow 2 段即归零);INT3 k=128 绷紧(全网并集 361 个
outlier 只有 128 个槽,溢出 233;DP 买 4 段)。如果分段在任何地方承重,就是这里。

| 臂 | 段数 | gather | seg 0 energy_cov | MEAN(8) | vs 同 init 父 run | train_runtime |
|---|---|---|---|---|---|---|
| segment P ×1 | 1 | **0** | L0–L31: **33.5%** | **78.43** | **−0.24** | 22750 s(6.487 samples/s) |
| **AutoSeg `[1,1,8,22]`(父,gptq_latent)** | 4 | 3 | 86.3 / 76.4 / 47.0 / 31.0% | **78.67** | — | —(4 卡) |
| AutoSeg(rtn init,T4b 主行) | 4 | 3 | 同上 | **78.85** | — | —(4 卡) |
| per-layer P ×32 | 32 | 31 | — | **78.43** | **−0.24** | 23020 s(6.410 samples/s) |

#### 关键:两格合起来,段数**不是单调旋钮**

| cell | ×1 | AutoSeg | ×32 |
|---|---|---|---|
| INT2 g32(k=256,DP 买 2 段) | 74.43 | **74.76** | **75.02** |
| INT3 g64(k=128,DP 买 4 段) | 78.43 | **78.67** | **78.43** |

INT2 看着单调上升,INT3 里 **×32 与 ×1 落在同一个数(78.43,两位小数完全相同)**,而 AutoSeg
在两者之上。**INT2 那条「越多段越好」的排序因此被证伪了**,它就是 ±0.58 噪声带里的一次抽样
——这也正是 DP 成本曲线早就预告的:overflow 在 2 段(k=256)/ 4 段(k=128)就归零,×32 在
AutoSeg 所优化的目标上一分钱没多买,现在在精度上也没买到。

**两格同向,幅度相近**:INT2 ×1 −0.33、INT3 ×1 −0.24(都对同 salient_init 的父 run)。
单看任一格都在 ±0.58 内不显著,但**两次独立的同号小幅下降**,加上机制侧的解释
(Jaccard 衰减 + gap 定位在 L0/L1 + DP 恰好切在那里),合起来是一致的图像:
**分段是真的在做事,只是只作用在 2–3 层上,所以总量小。**

**这对叙事是好消息**:主结果全部跑在 AutoSeg 下,而 ×1 在两个比特宽度上都只低 0.2–0.3,
可以老实写成「若推理栈无法承载 boundary gather,单段配置的代价是 0.2–0.3 分(在噪声带内)」——
是优雅降级,不是竞争配置。

### 各臂的置换结构(INT2 g32 bcal, k=256, 建基座时实测)
energy_cov = 该段 salient 并集在 256 列预算内覆盖的二阶矩能量比例。

| 臂 | boundary_sizes | 各段 outlier 并集 / energy_cov |
|---|---|---|
| ×1 | [32] | L0–L31: 361 / **36.1%** |
| 默认(×2) | [2, 30] | L0–L1: 217 / **81.3%**;L2–L31: 228 / 36.0% |
| ×32(per-layer) | [1]×32 | L0: 111 / **87.6%**;L1: 159 / 79.8%;L2: 105 / 64.0% … L31: 74 / 30.0% |

**×1 打平了默认:74.43 vs 74.76,差 0.33。** ×1 那一段的 energy_cov 36.1% 与默认第二段的 36.0%
几乎相同——默认多花的那一次 gather,买到的全部东西就是把 L0–L1 的覆盖率从 36% 抬到 81.3%,
而这两层值 0.33 分。也就是说在 INT2 g32 这一格,**分段机制(DP + per-segment P + 运行时 gather)
基本没有挣到它的复杂度**;真正干活的是 permute 本身(把 salient 搬到组边界上),不是"按段分开搬"。
×4 因此暂不跑:如果 per-layer(覆盖率上限)也打平,中间点不会有别的故事。

### AutoSeg 为什么站得住(全部零 GPU 成本的证据)
从 `seg32` 那份 base 的 meta 里直接读出 DP 自己的成本曲线(`auto_segments.cost_curve`,
commonsense bcal 3500 / INT2 g32 / group_k=256 / sigma=2.5):

| 段数 | 1 | **2** | 3 | 4 | … | 32 |
|---|---|---|---|---|---|---|
| overflow(预算外的显著通道数) | **105** | **0** | 0 | 0 | … | 0 |

**2 段就把 overflow 打到 0,再加段买不到任何东西。** 所以 DP 的选择不是调出来的,是它所声明的
目标函数的**精确最小值中段数最少的那个**——先验选择,没有看过任何下游精度。

三个 cell 的选择,同样是直接读出来的:

| cell | k | boundary_sizes | gathers |
|---|---|---|---|
| CS INT2 g32 bcal | 256 | `[2, 30]` | 1 |
| CS INT3 g64 bcal | 128 | `[1, 1, 8, 22]` | 3 |
| MetaMath INT3 g64 bcal | 128 | `[1, 1, 10, 20]` | 3 |

**k 越紧买的段越多,而且三格都把 L0 / L0-L1 单独切出来** —— 与上面 Jaccard 分析独立定位出的
「gap 集中在 L0(+4.0)/ L1(+4.4)/ L31(+2.3)」完全吻合。一个不知道 Jaccard 分析存在的 DP,
挑中了 Jaccard 分析指认的那几层,这是**预测-验证**,不是事后解释。

**这条曲线还顺手解释了 x32 的 +0.26 不是覆盖率带来的**:overflow 在 2 段就已经是 0,
x32 在 DP 的目标函数上一分钱也没多买。配合 ±0.58 的噪声带,它就是噪声。

**因此 AutoSeg 作为主配置的辩护是**:(1) 先验选择,未调参;(2) 在声明的目标下可证最小;
(3) 随 k 自适应;(4) 机制被 Jaccard 衰减 + gap 定位解释;(5) 成本实测 1-3 次 gather ≈ 0.04-0.12%
的 step time;(6) T7 的消融显示方法对段数**不敏感**——这是稳健性结论,不是弱点;seg1 则作为
「推理栈放不下 gather 时」的免费降级行(−0.33,在噪声内)。

**不要声称的**:AutoSeg 在精度上胜过 seg1(说不出显著性);分段是主贡献(它是机制,不是贡献);
以及 T3a 已修正的对 QEFT 的效率优势。

### 分段的动机(重写):相似性随层距离衰减,而不是「层间相似」
QEFT 的 OGR 已经占住了「salient index 跨层高度重合 → 用一个全局 P」这个动机(它自己的
docstring 就是这么写的)。所以我们的动机必须是它的**限定条件**,而这条限定是可量化的。
层的 outlier 集合(attn ∪ mlp)之间的 Jaccard,按层距离:

| 层距离 d | 1 | 2 | 4 | 8 | 16 | 31 |
|---|---|---|---|---|---|---|
| mean Jaccard | **0.725** | 0.651 | 0.548 | 0.421 | 0.319 | **0.163** |

**相邻层共享 72.5%,首尾两层只共享 16.3%。** 所以 QEFT 的前提**局部为真、全局为假**,
一个全局 P 必然亏待那些 salient 集合最有个性的层。

**但必须把第二半一起说:发散的通道不带能量。** 用每层自己的 top-k vs 全网 top-k:

| k | per-layer own | global(seg1) | gap |
|---|---|---|---|
| 256 | 26.6% | 25.4% | **+1.2 点(+5% 相对)** |
| 128 | 22.9% | 22.0% | +0.9 点(+4% 相对) |

**而且 gap 高度集中在三层**(k=256):L1 **+4.4**、L0 **+4.0**、L31 **+2.3**,中间层只有 +1.1~+1.6。
这正好解释了 DP 在干什么:INT2 选 `[2, 30]`(把 L0-L1 单独切出来)、INT3 选 `[1, 1, 8, 22]`
(把 L0、L1 各自单独切出来)。**DP 找的就是这几层,一个不多。** 也解释了为什么总体只值 0.33~0.59 分
——32 层里只有 2~3 层受影响。

### 一个 0-gather 的替代方案:单一全局 P + **逐层 k**
`group_k` 是**逐层 metadata,不是置换的一部分**(`layer_group_ks` 在
`group_k_for_module_name` 里已经是逐层读的,down_proj 早就在用),所以在**单一全局 P** 下给
难层更长的前缀,**不需要任何 boundary gather**。全局序下要多长才追平「该层自己的 top-256」:

| layer | own k=256 | 全局序需要的 k | 倍数 |
|---|---|---|---|
| L0 | 70.2% | **576** | 2.2× |
| L1 | 53.1% | **576** | 2.2× |
| L31 | 17.2% | **384** | 1.5× |

只对这三层加宽 ⇒ salient 参数 **+9.4%**,runtime gather **0 次**。

**关键不对称:k 对我们免费,对 QEFT 不免费。** 我们的 salient 列以部署位宽量化,加 k 不加
deployed bits;QEFT 的 weak 列是 FP16,加 k 直接加比特(k=256 就已经是 2.75 vs 2.00)。
所以「一个全局 P 亏待难层」这件事,**QEFT 只能用比特去补,我们可以用 k 去补**。

> ⚠ 这是**覆盖率**论证,不是精度测量。而覆盖率在本仓库已经被证明是弱预测器
> (seg1 25.3% vs seg32 26.8% → 74.43 vs 75.02,都在噪声内)。当作**值得跑一次的假设**,
> 不要当结论。实现成本很小:`layer_group_ks` 的下游管线已经全通,只需让选择阶段在
> `force_segments=1` 时输出逐层的 k 向量。

### 覆盖率扫描:段数 × k × 选择规则(CPU 实测,零 GPU 成本)
用 `salient_analysis_out/llama2-7b/second_moments.pt` 的 per-(layer, source) E[x²],
指标 = 全部 **64 个残差读点**(每层 attn / mlp 各一个)各自被自己那个 P 的 top-k 列捕获的能量比例,
取平均。`ovf` = 落在预算外的 sigma-outlier 总数。

| 配置 | gather | 规则 | k=256 | ovf | k=128 | ovf |
|---|---|---|---|---|---|---|
| seg1(单一全局 P) | **0** | 本仓库(并集+补齐) | 25.3% | 183 | 22.0% | 311 |
| seg1(单一全局 P) | **0** | QEFT(全局归一化排序) | **25.4%** | 183 | **22.0%** | 311 |
| seg32(每层一个 P) | 31 | 本仓库 | 26.6% | 0 | 22.9% | 14 |
| seg32(每层一个 P) | 31 | QEFT | 26.6% | 0 | 22.9% | 14 |
| seg64(每读点一个 P) | 63 | 本仓库 | **26.8%** | 0 | **23.1%** | 0 |
| seg64(每读点一个 P) | 63 | QEFT | 26.8% | 0 | 23.1% | 0 |

三条结论,都与直觉相反:

1. **选择规则不是差异来源。** 本仓库的「per-source sigma-outlier 并集 + 按分数补齐/裁剪」与
   QEFT 的「per-layer 归一化 λ 求和后全局 top-k」在同一预算下覆盖率只差 **0.1 个百分点**
   (25.3 vs 25.4),其余配置逐位相同。两者的统计量本来就是同一个(λ_j = diag(2XXᵀ) = 2T·E[x²]),
   聚合方式的差别在数值上被冲掉了。
2. **seg64 相对 seg32 只值 +0.2**,而且 **k 从 256 降到 128 并不会让它变得更值**(同样 +0.2)。
   原先「k 小了预算才绷紧、per-site 才有意义」的猜测**不成立**:k=128 下 seg32 的总 overflow
   也只有 14 个通道,共用 P 依旧塞得下两个读点的并集。
3. **k 256→128 是覆盖率降级**:seg64@k128 的 23.1% 低于 seg32@k256 的 26.6%,甚至低于
   seg1@k256 的 25.3%。同时 salient 可训权重从 314.6M 腰斩到 157.3M,而 T6 的 k 扫描里
   k=128 比 k=256 低 2.45 分(span 格 67.16 vs 69.61)。**「降 k 换 per-block 置换」在现有测量下
   是双输**,除非有别的证据。

> 口径同上:MetaMath 校准缓存(512×2048),不是本格的 commonsense bcal 3500;回答的是配置之间的
> 相对问题。

### step time:runtime gather 实测是免费的
| 臂 | gather 数 | s/it(单卡, per_device 4 × accum 20) |
|---|---|---|
| ×1 | 0 | 24900 s / 5.927 samples/s |
| ×32 | 31 | 25190 s / 5.859 samples/s(**慢 1.16%**) |

31 次 gather 只花掉 **1.16%** 的训练时间,即每次约 **0.037%** —— 比下面按带宽估的 0.4% 还便宜。

一次 boundary gather 是残差流上的 `index_select`,搬运量 O(B·T·d);它夹在其中的 decoder layer
是 O(B·T·d²) 的 GEMM。B=4、T≈256、d=4096、bf16 下:每次 gather 读写 16.8 MB,按 ~650 GB/s
实测带宽约 26 µs,反向的 scatter 约 60 µs,合计 ~86 µs;而实测每层每 micro-batch 约 20 ms
(13 s/step ÷ 20 accum ÷ 32 层)。**一次 gather ≈ 它后面那一层的 0.4%**,所以哪怕每层一次也看不见。
额外的 31 次 kernel launch(~155 µs/micro-batch)同样不构成瓶颈。

### 「per-layer」其实只有 32 个 P,不是 64 —— 以及为什么不补
残差流在每个 decoder layer 里被读**两次**:attention 读 `input_layernorm(x)`,MLP 读
`post_attention_layernorm(h)`。真正的 per-site oracle 应该给这两个读点各一个 P,32 层 = **64 个 P、
63 次 gather**。当前实现做不到,而且是三处一起卡住的:

1. `_segment_sources(start, end)` 把 `(l,"attn")` 和 `(l,"mlp")` 一起并进同一段的 outlier 并集 ——
   段的粒度就是「层」;
2. `_apply_residual_perm_fp32` 用同一个 `P_k` 同时置换 `input_layernorm` **和**
   `post_attention_layernorm`、q/k/v **和** gate/up 的输入列、o_proj **和** down_proj 的输出行;
3. `BoundaryGatherHook.register` 挂的是 `decoder_layer.register_forward_hook`,只能在**整层之后**触发。
   HF 的 `LlamaDecoderLayer.forward` 里 `residual = hidden_states` 是在 post-attn LN **之前**取的,
   所以在 LN 上挂 pre-hook 只会换掉 LN 的输入、换不掉后面要加回去的 `residual` —— 加法会基不一致。
   要做 64 段必须改写 decoder layer 的 forward(或包一层 wrapper),并把 fold 拆成 attn 侧 / mlp 侧两套。

**但先量了一下它值多少,结论是不值得。** 用 `salient_analysis_out/llama2-7b/second_moments.pt`
(Llama-2-7B 的 per-(layer, source) E[x²] 缓存)在 CPU 上按同一套 sigma-outlier + rank-fill 规则算
group_k=256 下的能量覆盖(共用 P vs 每个读点各一个 P):

| | attn 读点覆盖 | mlp 读点覆盖 | 平均 |
|---|---|---|---|
| 共用 P(= 现在的 seg32) | 38.1% | 15.1% | 26.6% |
| 各自的 P(真 64-P oracle) | 38.2% | 15.3% | **26.8%** |

**拆开只多 +0.20 个百分点。** 原因是 k=256 这个预算**根本没有绷紧**:seg32 自己的建基座日志里
每层 outlier 并集是 111 / 159 / 105 / … / 70 / 75 / 74,全部远小于 256,所以 attn 和 mlp 的
outlier 就算只有 56% 重叠(实测层内 Jaccard 0.563),两边的 outlier 也都能塞进同一个 256 里。
**只有当 group_k 小到装不下两个读点的并集时,64-P 才会开始有意义。**

> 口径说明:上表用的是 MetaMath 校准缓存(512×2048),不是本格的 commonsense bcal 3500;
> 绝对数值与日志里 `energy_cov` 的聚合口径也不同。它回答的是「拆 vs 不拆」的相对问题,
> 而支撑结论的那个结构事实(每层 outlier 数 ≪ group_k)在本格自己的日志里同样成立。

### 三个臂在统计上分不开 —— 不要做成「高配 / 低配」两档
1 → 32 段(置换数 ×32)总共只挪动 **0.59 分**,而这个差本身就在噪声里:

| 噪声来源 | 量级 |
|---|---|
| MEAN(8) 的二项抽样 SE(单臂) | ±0.41 |
| **两个独立臂之差的 1σ** | **±0.58** |
| T9 实测:同一 checkpoint 评两次的漂移 | 最大 0.34 |
| run-to-run 训练方差 | 未测(只会更大) |

**+0.59 ≈ 1σ,而且逐任务符号是混的**:WinoGrande +2.45、SIQA +1.23、OBQA +1.60 支持 ×32,
但 BoolQ **−1.01**、ARC-c −0.34、ARC-e −0.21 反向。如果 per-layer 置换真的多保护了显著能量,
应该是一致方向的小幅提升,而不是 8 个任务里 3 个倒着走。这是噪声的样子。

因此**不要把 ×32 和 ×1 包装成「高配 / 低配」两个方案**:那个说法预设了一条精度-成本折中曲线,
而本格的测量给不出这条曲线——它给出的是「三个点在误差棒内重合」。卖一个要 31 次无法折叠的
运行时 gather 的「高配」,换来一个说不出显著性的 +0.26,审稿人问一句就塌了。把 ×32 的 75.02
当主结果还有第二个问题:它是**三次含噪抽样里的最大值**(默认的 ×2 是 DP 先验选出来的,不是挑
出来的),报最大值是 forking-path。另外 75.02 仍低于 T1b 的 QEFT k=256+可训 zp 75.74,连本格
都没赢下来。

**这一格能诚实地说的是**:段数在 1→32 的范围内不影响精度,所以干活的是 **permute 本身**
(把 salient 搬到量化组边界上),不是**按段分开 permute**。要动主配置的话方向是 **×1 而不是 ×32**
——×1 是唯一 **0 次 runtime gather** 的臂,置换 100% 离线折进权重,部署产物就是一个普通的量化
checkpoint,不需要任何自定义算子,AGENTS.md 不变量 5(训练/导出/评测三处都要注册 boundary
gather)直接消失。但只凭这一格(INT2 g32 bcal,单 seed)不足以改主配置;要改,先在 INT3 g64
和另一个数据集上复现这个「不敏感」结论,或者多跑 seed 把 ±0.58 压下去。

**结论(对成本叙事重要):AGENTS.md §2 里"P_k 唯一无法离线折叠的部分"这项开销,在训练尺度上
实测约 0.4%/边界,T7 的 step time 一列是平的。**段数应当只按精度选,不必为运行时开销省。**
(限定:这是训练态、开梯度检查点、B=4、T≈256。推理态 batch/context 不同时绝对占比会变,
但它仍然按 1/d 缩放,量级不变。)

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
| MetaMath INT3 g64 格 | SALT-Q **zp×2 已完成并作为主行**(58.30 / 12.56,见 T2a2);QA-LoRA 对照待跑 |
| Wiki2 INT3 g64 格 | **已完成**,见 T2b2 |
| 第二模型(Llama-3-8B / Qwen2.5-7B) | 未跑 |
| 3 seeds std | 未跑 |
| Peak memory 实测 / kernel microbench | 未跑 |
| T5b:FP16 不训 + dense z-trained | 未跑 |
| T6:k=512、random-k 对照 | 未跑 |
| T7:per-site(64-P)、segment P ×4、无 P | ×1 已完成、per-layer(32-P)跑中;×4 暂不跑;64-P 未实现(需改 decoder forward,实测只值 +0.2 覆盖);无 P 未跑 |
