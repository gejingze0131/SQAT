# SALT-Q Experiment Tracker
Llama-2-7B · 统一 pipeline · final checkpoint · 3 seeds(主行) · `—` = 待跑

## T1 主表:Commonsense-170k → 8 bench(全宽)

### INT3 g64
| Method | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | Gap |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 upper | 69.3 | 81.1 | 77.5 | 92.4 | 76.6 | 82.7 | 64.4 | 78.0 | 77.75 | 100% |
| GPTQ floor | 66.9 | 77.4 | 72.7 | 86.2 | 69.1 | 76.4 | 60.5 | 71.6 | 72.60 | 0% |
| QA-LoRA | 68.9 | 79.5 | 78.1 | 91.1 | 75.9 | 78.7 | 65.1 | 77.2 | 76.82 | 82% |
| LoTA-QAF | — | — | — | — | — | — | — | — | — | — |
| QWHA | — | — | — | — | — | — | — | — | — | — |
| **SALT-Q k=128** | 70.5 | 81.4 | 78.1 | 91.3 | 78.8 | 80.6 | 65.4 | 76.0 | **77.76** | **100%** |

### INT2 g32
| Method | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | Gap |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 upper | 69.3 | 80.7 | 77.5 | 92.6 | 76.1 | 82.1 | 65.4 | 78.2 | 77.75 | 100% |
| GPTQ floor | 62.1 | 49.3 | 37.4 | 14.2 | 48.6 | 29.3 | 25.7 | 26.6 | 36.64 | 0% |
| QA-LoRA | 64.6 | 73.0 | 73.5 | 76.9 | 67.0 | 65.8 | 52.7 | 63.4 | 67.11 | 74% |
| LoTA-QAF | — | — | — | — | — | — | — | — | — | — |
| QWHA | — | — | — | — | — | — | — | — | — | — |
| **SALT-Q k=256** | 64.6 | 74.0 | 74.2 | 81.1 | 70.2 | 69.4 | 53.9 | 69.4 | **69.61** | **80%** |
| SALT-Q k=128, 3 ep (budget point) | 65.4 | 76.8 | 76.6 | 82.4 | 72.4 | 70.0 | 51.5 | 69.8 | 70.61 | 83% |

### ⚠️ 校准缺陷与 bcal(balanced-calibration)重建格 — 2026-08-29
上面两张 T1 表里所有 GPTQ 基座(SALT-Q 非显著 codes + 显著列/分段统计、QA-LoRA 基座、GPTQ floor)
都用训练文件前 N 条记录校准 = **100% BoolQ、9.5k 有效 token**(`calibration_sampling=first`)。
同一 QLoRA merged checkpoint 只换校准集(results_saltq.csv, INT2 g32 span):

| 校准 | tokens | MEAN(8) | 探针 ans_loss / ans_acc |
|---|---|---|---|
| 旧 floor(BoolQ-only 128) | 9.5k | 36.64 | 1.17 / 0.41 |
| C balanced 128 | 17k | 59.33 | 0.88 / 0.59 |
| B BoolQ-only 3500 | 256k | 44.93 | 1.15 / 0.45 |
| **D balanced 3500** | 471k | **66.22** | 0.72 / 0.67 |
| E C4 128×2048(标准) | 262k | 30.67(模板丢失) | 1.10 / 0.45 |

任务混合是主因(+24),预算其次(+7);C4 通用校准在 INT2 丢模板 → 保持 in-domain balanced。
**结论:旧 INT2 "Gap" 列作废**(正确校准的 PTQ floor 66.22 已与 1-ep SALT-Q k=128 / QA-LoRA 持平)。
INT3 floor 的 D/E 重跑(job 15762250):D = 76.05(旧 72.60),E(C4)= 72.07。INT3 也是校准假象:正确校准的 PTQ floor 距 QA-LoRA 0.8、距 SALT-Q 1.7。

**bcal 格(recipe 与 span 格逐字节相同,仅 `calibration_sampling: balanced`, `calibration_samples: 3500`,
`gptq.nsamples: 3500`;基座全部重建)**,configs/jobs 后缀 `_bcal`:

#### INT3 g64 · bcal
| Method | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | Gap | job |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 upper(不依赖校准,沿用) | 69.3 | 81.1 | 77.5 | 92.4 | 76.6 | 82.7 | 64.4 | 78.0 | 77.75 | 100% | — |
| GPTQ floor(D, balanced 3500) | 67.2 | 79.1 | 76.6 | 91.3 | 74.7 | 80.5 | 63.8 | 75.2 | 76.05 | 0% | ✅ |
| QA-LoRA | 69.3 | 80.1 | 79.2 | 92.1 | 77.3 | 82.8 | 64.3 | 76.6 | 77.71 | 98% | ✅ |
| QEFT k=128(论文主表设置;OGR 全模型单一置换 + Hessian 对角选列;**3+fp16**,不是纯 INT3) | — | — | — | — | — | — | — | — | — | — | ⏳ 15815171 → 待提交 |
| └ QEFT 自身 floor(裸基座:our GPTQ + balanced 3500,k=128 列留 fp16,**未微调**) | — | — | — | — | — | — | — | — | — | — | ⏳ |
| **SALT-Q k=128** | 71.1 | 81.7 | 78.8 | 92.2 | 78.6 | 83.1 | 65.9 | 79.4 | **78.85** | **164%** | ✅ |
| SALT-Q k=128, **salient_init=gptq_latent**(INT2 最佳定义的 INT3 复刻;lr 不变;15777648 因 builder bug 作废) | 70.6 | 81.4 | 79.5 | 92.2 | 78.3 | 83.5 | 65.8 | 78.0 | 78.67 | 154% | ✅ 与 RTN 起点持平(−0.18) |
| QEFT(paper lr 5e-6;eval 需 TP=4,TP=2+bf16 触发 vLLM sampler 崩溃) | 64.0 | 76.3 | 64.4 | 79.1 | 63.2 | 74.5 | 55.1 | 57.2 | 66.73 | −(低于本行自身 76.05 floor 之下的 raw-base 线,欠训练) | ✅ |
| QEFT,lr ×10 = 5e-5(eval TP=4) | 69.5 | 80.2 | 77.3 | 91.9 | 77.1 | 82.2 | 65.6 | 77.6 | **77.68** | 96% | ✅ paper lr 欠训练 10 分,×10 后与 QA-LoRA 持平 |

#### INT2 g32 · bcal
| Method | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg | Gap | job |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 upper(不依赖校准,沿用) | 69.3 | 80.7 | 77.5 | 92.6 | 76.1 | 82.1 | 65.4 | 78.2 | 77.75 | 100% | — |
| GPTQ floor(D, balanced 3500) | 58.8 | 71.8 | 71.9 | 78.9 | 65.0 | 68.8 | 53.1 | 61.6 | 66.22 | 0% | ✅ |
| QA-LoRA | 66.1 | 78.8 | 76.5 | 86.5 | 73.0 | 73.8 | 57.8 | 71.2 | **72.96** | **58%** | ✅ |
| QEFT k=256(OGR 全模型单一置换 + Hessian 对角选列;**2+fp16**,不是纯 INT2) | — | — | — | — | — | — | — | — | — | — | ⏳ 15815172 → 待提交 |
| └ QEFT 自身 floor(裸基座:our GPTQ + balanced 3500,k=256 列留 fp16,**未微调**) | — | — | — | — | — | — | — | — | — | — | ⏳ |
| **SALT-Q k=256** | 65.0 | 76.6 | 75.6 | 81.8 | 71.3 | 70.0 | 53.3 | 69.0 | 70.32 | 36% | ✅ |
| **SALT-Q k=256, salient_init=gptq**(显著列从全矩阵 GPTQ 解起步) | 66.1 | 78.5 | 76.0 | 87.3 | 73.3 | 76.3 | 59.7 | 74.0 | **73.91** | **67%** | ✅ |
| SALT-Q k=256, salient_init=gptq, **zp_lr ×2**(3.46e-3;同一冻结码基座) | 66.6 | 78.6 | 77.3 | 87.4 | 75.1 | 76.5 | 59.1 | 74.6 | **74.42** | **71%** | ✅ |
| SALT-Q k=256, salient_init=gptq, zp_lr ×4(6.92e-3;同一冻结码基座) | 66.2 | 77.8 | 76.1 | 86.4 | 74.7 | 73.0 | 56.3 | 74.6 | 73.13 | 60% | ✅ 过大(−1.29 vs ×2);zp_lr 定为 ×2 |
| SALT-Q k=256, salient_init=gptq, zp ×2 + salient_lr ×2(2.5e-4;同一冻结码基座,最后一臂) | 66.7 | 77.7 | 77.5 | 87.0 | 75.1 | 75.3 | 59.0 | 76.4 | 74.36 | 71% | ✅ 持平(−0.06);salient_lr 保持 1.25e-4 |
| SALT-Q k=256, **salient_init=gptq_latent**(95.7% 显著权重保留 fp16 原值、4.3% 落 GPTQ 格中心;codes/网格/lr 同 zp×2) | 66.4 | 77.9 | 77.7 | 87.7 | 76.5 | 76.1 | 59.1 | 76.8 | **74.76** | **74%** | ✅ 当前 INT2 最佳 |
| QEFT k=256(paper-law lr 3.5e-6;TP=4 补评) | 62.1 | 69.2 | 58.5 | 46.5 | 53.0 | 65.8 | 44.5 | 50.4 | 56.26 | −(欠训练,低于 floor) | ✅ |
| QEFT k=256,lr 5e-5(eval TP=4) | 65.5 | 77.9 | 76.8 | 87.6 | 73.3 | 75.6 | 59.6 | 73.2 | **73.71** | 65% | ✅ 高于 QA-LoRA 72.96,低于 SALT-Q 74.76 |
| SALT-Q k=256, gptq_latent, **zp 冻结**(zp_lr=0;QEFT 镜像:只训显著列+scale) | 65.7 | 77.4 | 75.8 | 86.6 | 74.0 | 74.7 | 59.7 | 74.8 | 73.59 | 64% | ✅ zp 学习值 +1.17;无 zp 时与 QEFT 73.71 打平(纯 2bit vs 2+fp16) |
| QEFT k=256,lr 5e-5 + **可训 zp**(zp_lr 3.46e-3;补全 2×2) | 66.3 | 79.4 | 77.5 | 89.0 | 78.4 | 75.5 | 60.6 | 79.2 | **75.74** | **83%** | ✅ INT2 新最高;zp 增益 +2.03(QEFT 侧)/+1.17(SALT-Q 侧) |

| **LoTA-QAF**(论文主表设置 ω=0.75r=48, σt 5%→0.1%→0.01%;bcal matched base,逐位同格) | 62.2 | 63.7 | 51.1 | 27.3 | 51.1 | 57.8 | 40.8 | 46.4 | **50.04** | 60%※ | ✅ 15775273 |
| └ LoTA-QAF 自身 floor(裸基座:our GPTQ + balanced 3500,**未微调**) | 5.4 | 13.4 | 15.8 | 5.8 | 8.8 | 4.3 | 3.3 | 13.0 | 8.72 | 0% | ✅ |

※ **LoTA-QAF 的 Gap 与本表其它行不同基准,不能并排读**:它按自身 floor 8.72 算
((50.04−8.72)/(77.75−8.72)),而 QA-LoRA/SALT-Q 的 Gap 按 GPTQ floor D 66.22 算。两者差别是
**种类**上的:66.22 是 QLoRA 微调后再量化的模型(已经会输出答案格式),8.72 是从未微调过的原始
Llama-2 量化后的模型(几乎无法产生可解析答案)。可直接比较的是**绝对分**:LoTA-QAF 50.04 vs
QA-LoRA 72.96 vs SALT-Q 74.76 — 三者同基座、同数据、同 batch、同 1 epoch。

**为什么低这么多(已量化,非猜测)**:ω=48 下整数合并几乎不发生 —— 抽查 4 层 q_proj(6700 万权重)
在训练结束后总共只有 **6 个** marker 越过阈值(INT3 那次是 4621 个)。|AB| 是 r 个三值乘积之和,
std=√(4r/9)=5.3 @ r=64,ω=0.75r 是约 9σ 的门槛。所以全部提升都来自 offset factor,而**发布代码把
它除以了 ω**(`layer.py:359`/`adapter.py:391`/`lota_merge.py:382`),论文 Eq.(4) 没有这个除法:
实测该 checkpoint 上 |μ| = 0.022 个量化步长,按论文公式则是 1.06 个。三条路径(训练/推理/合并)除法
一致,所以不是 train/deploy 不一致,本行忠实于发布代码;Eq.(4) 那一臂未跑。

LoTA-QAF / QWHA / QEFT 的对照臂应在 bcal 基座(或同等 balanced 校准)上跑,否则与本格不可比。

**QEFT 行怎么读(baseline/QEFT/sqat/PROVENANCE.md)**:QEFT 把每层 k 列保留 **fp16** 并且只训练这
k 列,其余冻结,所以它和 QWHA 一样**不是纯 INT-b 行**——部署形态是 INT-b codes + k 列 fp16
(k=128 → 目标权重的 2.69%、3.35 等效 bit;k=256 g32 → 5.37%、2.75 等效 bit)。它应该与
**同 k 的 fp16-salient PTQ 扫描点**对照着看,但**两者不是同一构造**:PTQ 扫描是把已经
QLoRA 微调过的 merged checkpoint 量化(`export_mixed_precision_sweep.py`),QEFT 与 QA-LoRA /
SALT-Q / LoTA-QAF 一样从**预训练 Llama-2-7B** 起步,所以 `--with_base` 的自身 floor 是一个从没见过
指令格式的模型,应当落在 LoTA-QAF floor(8.72)那个量级,而不是 66–77。训练后的分数按绝对值与本格
其它行并排读;floor 只作为它自己 Gap 的分母。与 SALT-Q 的三点差异:全模型共用一个置换(SALT-Q 按
segment,且需要 boundary gather)、显著列走 fp16 直训而非 INT-b QAT、选列用 Hessian 对角
λ_j = diag(2XXᵀ) 的逐层归一化全局投票而非 per-segment 的 E[x²] union。
(λ_j 与 E[x²] 其实是同一个统计量差一个常数 2T;真正的差别在聚合方式与"全局 vs 分段"。)

**INT2 bcal 诊断(2026-08-29)**:训练末 5% loss SALT-Q 1.218 vs QA-LoRA 1.220(相同),但测试 70.32 vs 72.96,差距集中在
HellaSwag −4.7 / ARC-c −4.5 / ARC-e −3.8(知识型任务)。差别在起点:SALT-Q 首个 loss **5.80**(旧基座 4.61),QA-LoRA **2.47**。
原因:SALT-Q 的显著列在 step 0 是 min-max 网格上的 RTN(GPTQ 把它们钉在 canonical grid,不做 OBS 补偿;非显著块又是独立求解、
不补偿显著列误差),INT2 g32 下对能量最高的 256 列做 RTN 是灾难性的;balanced 统计把更多真实能量选进显著列
(Seg0 81%/Seg1 36% vs 旧 43%/26%),所以起点反而更差。QA-LoRA 的 step 0 = 全 GPTQ(= floor 66.22)。
1 epoch 的任务数据能把训练 loss 拉平,但拉不回被 RTN 摧毁的知识。不是学习率问题(lr 括号在 ±1 内平坦;训练 loss 已与 QA-LoRA 持平)。
INT3 不受影响(首 loss 2.38,RTN 损伤小)。修正方向:显著列用 GPTQ 解出的 (W_int,s,z) 作为 LSQ 起点(step 0 = 全 GPTQ 模型),
其余 recipe 不变。
→ **已验证(job 15770332)**:首 loss 5.80 → 2.62,MEAN(8) 70.32 → **73.91**(QA-LoRA bcal 72.96),OBS 移动了 4.51% 的显著 codes;`salient_init: gptq` 成为 INT2 默认推荐。

## T2 副表:GSM8K + MATH500 + WikiText-2 PPL(单栏)
GSM8K/MATH500: MetaMath 微调;Wiki2: 在 train split 微调报 test PPL(LoftQ/ApiQ 协议)

| Method | Bits | g | GSM8K↑ | MATH500↑ | Wiki2 PPL↓ |
|---|---|---|---|---|---|
| FP16 upper | 16 | — | 45.2 | — | — |
| GPTQ floor | 3 | 64 | 28.7 | — | — |
| QA-LoRA | 3 | 64 | 39.0 ⚠️复核tuned-lr | — | — |
| LoTA-QAF | 3 | 64 | — | — | — |
| QWHA | 3 | 64 | — | — | — |
| **SALT-Q** | 3 | 64 | **45.11** | — | — |
| GPTQ floor | 2 | 32 | 崩 | — | — |
| QA-LoRA (tuned) | 2 | 32 | 37.38 | — | — |
| LoTA-QAF | 2 | 32 | — | — | — |
| QWHA | 2 | 32 | — | — | — |
| **SALT-Q** | 2 | 32 | **40.94** | — | — |
| **SALT-Q**(span bcal gptq_latent,CS 经验 lr,zp×2,balanced 1k 校准) | 2 | 32 | **56.33**(旧格 40.94) | — | — |
| QLoRA fp16 上限(同 recipe,lr 2e-4,span) | 16 | — | **58.07**(MATH 10.64) | — | — |
| QLoRA→GPTQ floor(58.07 merged ckpt 经 balanced 1k 校准 GPTQ INT2 g32) | 2 | 32 | 15926927 排队(15895788 在 g3 Q 了 14h "Insufficient node_pool",qdel 后改 3h walltime 重投以便 backfill) | — | — |
| QA-LoRA(同 recipe,lr 5e-3,balanced 1k 基座) | 2 | 32 | **52.54**(MATH 11.04;train_loss 0.2646) | — | — |
| QEFT(fp16 弱列,无 zp;weak lr 5e-5) | 2 | 32 | **48.90**(裸基座 0.00) | — | — |

## T3 Pareto 表:vs 全量 QAT(精度+成本,全宽)
1×A100-80G,seq/batch 统一,吞吐相对 QLoRA=1.00;显存/时长必须实测,禁止引原文

| Method | Bits | g | CS-Avg | GSM8K | Peak Mem | Tok/s(rel) | Time/ep | W mat.? |
|---|---|---|---|---|---|---|---|---|
| QLoRA (cost anchor) | 4+16 | 64 | — | 45.2 | — | 1.00 | — | No |
| QA-LoRA | 3 | 64 | 76.82 | 39.0 | — | — | — | No |
| LR-QAT | 3 | 64 | — | 41.4 | — | — | — | Yes |
| EfficientQAT | 3 | 64 | — | — | — | — | — | Yes(bw) |
| **SALT-Q** | 3 | 64 | **77.76** | **45.11** | — | — | — | No |
| QA-LoRA | 2 | 32 | 67.11 | 37.38 | — | — | — | No |
| LR-QAT | 2 | 32 | — | — | — | — | — | Yes |
| EfficientQAT | 2 | 32 | — | — | — | — | — | Yes(bw) |
| **SALT-Q** | 2 | 32 | **69.61** | **40.94** | — | — | — | No |

## T4 消融:adapter vs native fields(salient QAT + P 固定)
| Arm | INT3g64 GSM8K | INT2g32 GSM8K | INT2g32 CS-Avg | Tok/s |
|---|---|---|---|---|
| (a) salient QAT only, 无P (scatter) | — | — | — | — |
| (b) +P, dense 静态 | — | — | — | — |
| (c) +BF16 LoRA 合并 | 42.6 | — | — | — |
| (d) +z-training = **SALT-Q** | **45.11** | **40.94** | **69.61** | — |
| (d') z-only(无salient QAT) | — | 37.95 | — | — |
| (e) +s&z | — | — | — | — |
| (f) s-only, 无salient无P(≈PEQA) | — | — | — | — |

## T5 解耦 2×2:bits vs gradients(INT3 & INT2 各一组)
| Salient \ Dense | dense 静态 | dense z-trained |
|---|---|---|
| FP16 可训(QEFT式) | — | — |
| FP16 不训(PTQ保护,已有) | INT3:74.72 / INT2:58.96 | — |
| FP16 不训,**bcal 基座**(rank-ordered top-k,balanced 3500 Hessian) | INT3: 76.42@k64(3.16b) … 76.96@k2048(8.05b),floor 76.05,曲线在噪声内平坦 / INT2: 69.68@k32(2.09b) … 70.65@k512(3.36b) … 72.57@k2048(7.44b),floor 66.22:fp16 显著列一上来 +3.5,之后到 10% 份额几乎不再涨 | — |
| 同bit QAT(ours) | — | bcal: INT3:**78.85**(RTN 起点)/ 78.67(gptq_latent) / INT2:**74.76**(gptq_latent, zp×2) |

注:附 eff. bits 列;"FP16可训+dense静态"是真正的 QEFT 哲学,必补

## T6 salient 预算 k 扫描
| k | INT3g64 CS-Avg | INT2g32 CS-Avg | INT2g32 GSM8K |
|---|---|---|---|
| 64 | — | — | — |
| 128 | 77.76 | 67.16 | 40.94 |
| 256 | — | 69.61 | — |
| 512 | — | — | — |
| random-128 对照 | — | — | — |

## T7 permutation 消融
| 配置 | INT3 CS-Avg | INT2 CS-Avg | step time |
|---|---|---|---|
| per-layer P(oracle) | — | — | — |
| segment P ×1 | — | — | — |
| segment P ×2 | — | — | — |
| segment P ×4 | — | — | — |
| 无P(scatter-gather) | — | — | — |

## 图
- F0: 列级 Fisher vs E[x²] 秩相关(256样本一次backward)
- F1: 相邻层 outlier 集合 Jaccard 曲线
- F2: FP16微调 ΔW 列能量 vs E[x²] recall@k
- F3: LSQ step size 漂移 / salient vs 非salient 位移
- Pareto 散点: INT2 CS-Avg vs Peak Mem

## 队列(优先级)
1. ~~主结果~~ ✅ → **bcal 格已出分**(INT3: SALT-Q 78.85 > QA-LoRA 77.71;INT2: SALT-Q k=256 70.32 < QA-LoRA 72.96 — 见 bcal 小节的诊断);待定:INT2 SALT-Q 显著列初始化修正后重算 T1/RESULTS_SUMMARY.md
2. LoTA-QAF / QWHA 复现(T1/T2;LoTA-QAF 有官方代码、GPTQ 系基座对齐成本低;ApiQ 不跑,引用+附录讨论,由 QWHA 代表初始化线)
3. trained FP16-salient(T5 关键格,与 QEFT 实现同源)
4. EfficientQAT / LR-QAT(T3:精度+实测成本)
5. Wiki2 + MATH500 补列
6. 第二模型(Llama-3-8B 或 Qwen2.5-7B)INT3+INT2 主设置 ← reviewer 硬需求
7. T4/T6/T7 补格;QA-LoRA INT3 tuned-lr 复核
8. 3 seeds std;效率实测;kernel microbench
9. (stretch) 13B / MMLU
