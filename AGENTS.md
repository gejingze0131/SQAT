# AGENTS.md — SQAT 仓库工作指南 & SALT-Q

# NSCC 环境

- 用户目录：`/home`
- Scratch：`/scratch`
- Conda：`~/miniforge3`
- 默认环境：`saltq`

激活环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate saltq
```

大型模型、数据集、checkpoint 和缓存应放在 Scratch，不要占用 Home。常用缓存路径：

```bash
export HF_HOME=/scratch/cache/huggingface
export TORCH_HOME=/scratch/cache/torch
export PIP_CACHE_DIR=/scratch/cache/pip
export VLLM_CACHE_ROOT=/scratch/cache/vllm
export WANDB_DIR=/scratch/cache/wandb
```

NSCC 登录节点不要直接运行 GPU 训练。GPU 任务需通过 PBS `qsub` 获取计算节点后再运行 `accelerate` / `deepspeed`。

项目：`personal-jingzege`

GitHub SSH 可能被 NSCC 网络阻断；优先使用 HTTPS remote + PAT。

本文档有两部分：

- **第 I 部分**：现有 **Permuted SQAT** 方法的完整叙事与代码地图（agent 动手前必须先读懂的部分）。
- **第 II 部分**：面向 ≤2bit 的方法 **SALT-Q**（Saliency-Allocated Low-bit Trainability）的设计叙事、实现现状与后续计划。

> 写代码前的硬性要求：第 I 部分「不可破坏的不变量」一节里的每一条，都是过去踩过坑、修过 bug 才立下的。任何改动都不得违反；如果必须违反，先在 PR/commit message 里说明理由。

---

# 第 I 部分 · 现有方法：Permuted SQAT

## 1. 问题起点

QLoRA 的根本矛盾：训练时 base 是低比特（NF4）、adapter 是 BF16 的 `B@A`；部署时要么

- 保留 BF16 adapter（推理不是纯低比特，kernel 不统一、显存不省），要么
- 把 `W' = W_base + B@A` 合并后**重新量化**——而重量化会把训练好的 delta 大部分抹掉（"merge damage"）。

QA-LoRA 用「group-pooled adapter → delta 组内恒定 → 折进 affine zero-point」绕开了这个矛盾，但代价是 adapter 的表达力被压缩成"每 (行, 组) 一个平移量"且受 rank 限制。LR-QAT 把 LoRA 放进量化器内部（`Q(W0 + sBA)`），精度好但**全权重每步 fakequant**，训练开销大。

Permuted SQAT 的立场：**只对量化误差伤害最大的那 1–2% 参数做 QAT，其余交给 PTQ**。

## 2. 为什么要 permute

量化误差主要由 salient（高激活二阶矩）输入通道主导，但这些通道**分散在整个输入维度上**。group-wise 量化里，一个 salient 通道会污染它所在的整个 group 的 scale。

关键观察：**同一 segment 内相邻层的 salient channel index 高度相似**。于是：

1. 按层分 **segment**，每个 segment 共享一个残差流置换 `P_k`，把该 segment 内所有层的 salient channel 并集搬到物理位置 `[0, group_k)`；
2. `group_k % group_size == 0`，所以 salient 恰好占满**前 1–2 个量化组**（例：`group_size=64, group_k=128` → 前 2 组）；
3. 对这前 `group_k` 列做 QAT，其余列纯 GPTQ。

三个等价变换（全部离线折进权重，模型输出不变）：

| 变换 | 作用对象 | 代码 |
|---|---|---|
| (A) 残差流置换 `P_k`（按 segment） | q/k/v/gate/up 的输入列 + o/down 的输出行 + 两个 LN + embed/lm_head | `apply_segment_permutation_fp32` |
| (B) MLP 块内置换 `P4_l`（按层） | gate/up 输出行 + down 输入列 | `apply_block_internal_permutations_fp32` |
| (C) per-head Hadamard `H`（按层） | v_proj 输出行 + o_proj 输入列 | `apply_hadamard_rotation_fp32` |

`P_k` 唯一无法离线折叠的部分是 **segment 边界的残差重排**（skip connection 没有权重载体），必须在运行时做 `num_segments-1` 次 `index_select`：`BoundaryGatherHook` / `register_boundary_gathers_from_meta`。**训练、导出后的推理、lm-eval 三处都必须注册**，漏一处模型直接崩。

`o_proj` 没有连续 salient 切片（多头结构不允许跨头置换），因此只做 Hadamard，**不做 QAT**（`group_k_for_module_name` 对 o_proj 返回 0）。

## 3. 流水线（当前）

```
Stage A  build_permuted_fp16_checkpoint()            [rank0, 一次]
   载入 fp16 base → 采集 E[x²]（residual + down_proj）
   → 自动分段 (DP: auto_segment_by_outliers / auto_segment_with_fixed_group_k)
   → 选 salient 通道 → 构造 P_k / P4_l / H → 就地折叠
   → save_pretrained(permuted_fp16_base/) + sqat_permute_meta.pt
   （量化只发生一次：permute 在干净 fp16 上做，之后走标准 load_in_4bit）

Stage B  SegmentPermutedSelectiveQAT.prepare_model()  [训练]
   注册 boundary gathers
   → install_fused_selective_qat(): 块级融合 hook 注入
       attn : self_attn 的 forward_pre_hook 一次算出 q/k/v 的融合 delta
       mlp  : 同理 gate/up
       down : 单独一个小 GEMM
     delta = fakequant(W_base_S + (B@A_S)·scaling) − W_curr，只在 [0:group_k) 上
   → 挂 _sqat_permute_meta 供导出

Stage C  export.merge_and_export()                    [导出]
   NF4 dequant + LoRA merge → dense fp16（permuted basis）
   → salient 切片用 canonical/LSQ 网格；非 salient 用 GPTQ（顺序、逐层、带 boundary gather）
   → quantize→dequant 存成 dense fp16 + sqat_permute_meta.pt

Eval  lm_eval_model_kwargs() 自动构建带 boundary gather 的 HFLM
```

三个可选增强（都已实现、可组合）：

- **AWQ-S**（`awq_scale`）：salient 切片在放大空间 `W_S·S` 里量化，导出时把 `1/S` 烘回 dense 权重；纯粹是更好的网格，输出等价，无运行时开销。
- **GPTQ 非 salient**（`gptq.enabled`）：非 salient 列用 OBS 误差补偿代替 RTN。**salient 切片的量化误差绝不传播到非 salient 列**（训练时非 salient 是未重量化的 fp16，让 GPTQ 去"吸收" salient 误差反而会偏离训练时的输出）。
- **LSQ**（`lsq.enabled`）：salient 切片的 per-step min-max scale 换成可学习 `nn.Parameter`（asym 学 scale+zp），`current_minmax` 初始化。

## 4. 效果与失效点

真实结果来自 6.12 实验报告（Llama2-7B，MetaMath 395k 全量微调，GSM8k 5-shot exact-match，asym）。
**注意：`results/math/*.json` 里那批是上一轮开发的中间产物，不是 baseline，不要拿来对照。**
权威数字已落在 `baselines_metamath.csv`，由 `collect_saltq_results.py --seed` 并入结果表。

| Method | INT4 g64 | INT3 g64 | INT3 g32 |
|---|---|---|---|
| QLoRA (mix) fp16 参考 | **45.2** | 45.2 | 45.2 |
| QLoRA (merged, RTN / GPTQ) | 41.2 | 26.1 / 28.7 | 36.5 / 38.8 |
| Full STE-QAT (LR-QAT) | **45.2** | **45.2** | — |
| QA-LoRA | 42.8 | 39.0 | — |
| SQAT (top-1% / top-2%) | 43.7 / 43.8 | 40.7 | — |
| PSQAT+RTN | 43.6 | 40.8 | — |
| **PSQAT+GPTQ** | **45.2** | **42.6** | **44.8** |
| SalientFP16+GPTQ（salient 存 fp16） | — | 42.3 | 44.8 |
| **SALT-Q（本机 lm-eval）** | — | **44.8** | — |

> SALT-Q 的 44.8 是**实测**（strict-match 44.81 / flexible 45.11，`results_saltq.csv`
> `source=lm-eval`，2026-08-06 run2：INT3 g64、连续 z、z-only、salient_lr 2e-4、zp_lr 1e-3）。
> 它把 PSQAT+GPTQ 的 42.6 抬了 **+2.2 分**，与 LR-QAT / fp16 的 45.2 差 0.4 分（在 ±1.37 的
> 标准误内），也就是 §4 那个"2.6 分缺口"基本被吃掉，且成本仍在 PSQAT 量级。

三个必须记住的事实：

1. **LR-QAT 是精度上界；差距在效率，不在精度。** 修好实现后，LR-QAT 在 INT4 g64 **和** INT3 g64
   下都能完全恢复 fp16（45.2, 45.2）。PSQAT+GPTQ 在 INT4 g64 打平（45.2），在 INT3 g64 落后
   **2.6 分**（42.6）。代价一侧：LR-QAT 训练时间 **+175%**，PSQAT 只有 **+8.7%**。
   > 报告早期版本记录的 LR-QAT 3bit g64 = 41.4、初始 loss = 10.0 **是实现 bug**，已作废。
   > 不要再把"LR-QAT 在低比特不稳定"当作论据——它不成立。
2. **低比特 salient QAT 打得过把 salient 留在 fp16**：INT3 g64 下 PSQAT+GPTQ 42.6 > SalientFP16+GPTQ
   42.3；INT3 g32 下两者持平（44.8）。也就是说不必保留高精度通道。
3. **效率**（W3g64，bs=8×ga=2×4卡）：QLoRA 6.76h / 23.2 GB，PSQAT 7.35h（**+8.7%**）/ 24.2 GB，
   SQAT 7.95h（+17.6%）/ 26.3 GB，QA-LoRA 6.43h（−4.8%），LR-QAT 18.9h（**+175%**）/ 26.8 GB。

正确的叙事（报告结论 3，已取代旧版）：**对所有权重都做 QAT 是计算与内存的浪费；只需对 salient
channel 聚集的那一段做 QAT，剩下的 non-salient 段没有 outlier，用轻量的 GPTQ 就能恢复性能。**

**因此 SALT-Q 要证明的命题是**：在 PSQAT 量级的成本下拿到 LR-QAT 量级的精度。INT3 g64 的
2.6 分缺口就是它要吃掉的目标；2bit 则是 PSQAT 直接崩掉、而全权重 QAT 仍然可行的区间。

**失效点：≤2 bit。** 到 2bit，non-salient 段仅靠 GPTQ 已经压不住误差，而它在训练中
**完全没有任何自由度**（LoRA 在这些列上的 delta 导出时会被重量化抹掉——正是 §1 那个 merge 矛盾）。
这正是 SALT-Q 存在的理由。

## 5. 代码地图（`src/`）

M0 重构后，`qat_permute_sqat.py` 里与方法无关的部分已抽成三个共享模块（SALT-Q 全部复用）。
`qat_permute_sqat.py` 保留 **re-export shim**，所有旧 import 路径继续可用。

| 文件 | 内容 |
|---|---|
| `quant_primitives.py` | **canonical 量化网格的唯一真相源**：`_sym_scale` / `_asym_qparams`、`group_fakequant` / `group_quantize` / `group_dequantize`、LSQ 转发、`lsq_scale_for_module` |
| `permute_common.py` | 标定 `_collect_second_moments`、saliency 与分段 DP、P_k / P4_l / Hadamard、`BoundaryGatherHook`、`build_permuted_fp16_checkpoint`、perm-meta I/O、AWQ-S、lm-eval 胶水 |
| `gptq.py` | `gptq_quantize_layer`、`gptq_quantize_model_sequential`（Permuted SQAT 导出、QA-LoRA 基座、SALT-Q 冻结码字三方共用） |
| `qat_base.py` | `QATHandler` / `QATMode` / 工厂、LSQ 与 LSQ+ 原语、`grad_scale`、`FullQAT`（LR-QAT baseline） |
| `qat_permute_sqat.py` | Permuted SQAT 独有部分：融合注入器、`fused_qat_residual_outputs`、`SegmentPermutedSelectiveQAT` |
| `qat_saltq.py` | **SALT-Q 全栈**：`build_saltq_base`、`SALTQLinear`、`build_saltq_model`、可训张量存取、`SALTQ` handler、`export_saltq` |
| `qat_sqat.py` | 旧的 per-layer wrapper 版 SQAT（`sqat` 模式） |
| `qalora.py` | QA-LoRA：GPTQ INT-b 冻结基座 + group-pooled adapter |
| `export.py` | LoRA 系方法的合并 & 导出（SALT-Q 不走这里，它没有可合并对象） |
| `trainer.py` | HF Trainer 封装、`QATCallback`、LSQ / SALT-Q 的 optimizer 分组与 checkpoint 覆写 |
| `model_loader.py` | bnb NF4 + PEFT 装载（SALT-Q 不使用，见 `qat_saltq.build_saltq_model`） |

测试：`scripts/test_saltq.py`、`test_saltq_e2e.py`、`test_qat_permute_sqat.py`、`test_lsq_consistency.py`、
`test_lsq_permute_consistency.py`、`test_gptq_nonsalient.py`；等价性验证 `scripts/verify_permute.py`。
**改动任何量化/置换代码后，这 6 个套件必须全绿。**

## 6. 不可破坏的不变量

1. **训练网格 == 导出网格**。这是本仓库历史上最贵的一类 bug（min/max 与 pos/neg 两套 asym 公式混用）。
   - `quant_primitives._asym_qparams` / `_sym_scale` 是 **sqat_permute / SALT-Q 家族**的唯一真相源。
   - `qat_base.asymmetric_scale_zero_from_pos_neg` 是**另一套**（旧 `sqat` / AWQ 导出用）。**新代码不要混用。**
   - `qat_base` 的 LSQ 系列自成一套；**asym LSQ+ 在 `current_minmax` 初值下与 canonical affine 网格数值恒等**
     （同样的 `scale=(wmax-wmin)/q_max`、`zp=round(-wmin/scale)`、同样的 `(q-z)s`），这正是 SALT-Q
     "step 0 == GPTQ 重建" 成立的原因；symmetric LSQ 的 `Qn=-2^(b-1)` 与 canonical sym 的 `-q_max` **不一致**。
2. `grad_scale` 必须是**前向逐位恒等**的 autograd.Function。旧的 `(x - x*g).detach() + x*g` 只是代数恒等，
   fp32 下会差 1 ulp；训练 fakequant 对 scale/zp 用 grad_scale 而导出量化器用原始参数，这个 ulp 会直接变成
   train/deploy 的数值差（边界上甚至变成不同的整数）。SALT-Q 能断言 `max|Δ| == 0` 就靠这一条。
3. 量化取整**必须在 fp32 里做**（fp16 下小幅值会静默变成 no-op）。
4. `group_k % group_size == 0`；salient 切片是物理连续的 `[..., :group_k]`，不允许再做 index_select。
5. boundary gather 必须在训练 / 导出模型推理 / lm-eval 三处都注册。
6. Permuted SQAT 专有：绝不 materialize 完整的 `W + B@A`；绝不替换 BnB/QLoRA 的 projection forward；
   q/k/v 之间、gate/up 之间不共享 qparams；`lora_dropout` 必须为 0。
7. resume 训练必须复用原来的 `permuted_fp16_base`（SALT-Q 还要复用 `saltq_base` —— codes 是一次性离散选择，
   重新生成会让 checkpoint 里的每个张量全部错位）。

---

# 第 II 部分 · SALT-Q

## 7. 命名

**SALT-Q = Saliency-Allocated Low-bit Trainability**（也写作 SALTQ）。

名字直接编码方法本体：贡献重心不在"排列"，而在**可训练自由度的分配**——salient 段满自由度、其余仿射自由度、
codes 零自由度。permutation 只是让这个分配能落在连续的量化组边界上的手段。

- QAT 模式字符串：`"saltq"`　主模块：`src/qat_saltq.py`　配置：`configs/saltq.yaml`　启动：`runs/saltq/run_saltq_{math,commonsense}.sh`
- 输出目录：`outputs/saltq-{bits}bit-saltq/`（训练）、`outputs/saltq-{bits}bit-saltq-deploy-eval/`（导出）

## 8. 核心叙事

把原本要靠 LoRA delta `BA` 承担的任务适配，拆给两个**都以部署原生形态被训练**的段：

| 段 | 占比 | 部署载体 | 自由度 | 训练方式 |
|---|---|---|---|---|
| **Salient** `[0:group_k)` | 1–2% | INT-b codes | **权重空间满秩** | 权重本身是 `nn.Parameter`，LSQ fake-quant QAT |
| **非显著** `[group_k:)` | 98–99% | INT-b codes（**冻结**）+ metadata 槽位的 `(s, z)` | 每 (输出行, 量化组) 一个缩放 + 一个平移 | codes 冻结，只训 `(s, z)` |
| codes 本身 | — | — | **零** | — |

- `BA` 的**高秩成分**由 salient 段的直接权重训练承接；
- `BA` 的**组内平移/缩放成分**由 `(s, z)` 承接。

因此：salient 段训练结束时权重本来就落在量化格点上，merge 即取整；非显著段训好的 `(s, z)` 直接写进 packed
格式原有的 metadata 槽位，字节布局与普通 GPTQ checkpoint 完全相同。**整条流水线不存在需要合并的 BF16 对象。**

### 8.1 定位

- **QA-LoRA 是非显著段的退化特例**：其 group-pooled adapter 的 delta 组内恒定，部署时恰好折进 affine
  zero-point——即它训练的是 `z` 的一个 rank-r 受限参数化。SALT-Q 直接训每 (行, 组) 的 `z`（无 rank 约束），
  并且同时训 `s`。
- **LR-QAT** 全权重每步 fakequant 且仍需 BF16 低秩项；SALT-Q 只 fakequant 1–2%，非显著段完全不需要
  fakequant（前向只是一次 `(q−z)·s` 的 elementwise 重建）。**LR-QAT 是精度上界**（§4：3/4bit g64 都
  完全恢复 fp16），但训练时间 +175%。SALT-Q 的赌注是：把非显著段的自由度从"零"提升到"仿射"，
  足以补上那 2.6 分，而成本仍在 PSQAT 量级。
- **不再经过 NF4**：当前 2/3bit 走的是"NF4 基座 + 在 fakequant/导出里做真实位宽取整"
  （`model_loader._get_bnb_config` 的 WARN）。SALT-Q 的基座就是 INT-b codes，训练看到的就是部署的网格。
- **o_proj 首次可训**：它 `group_k=0`，在 SALT-Q 里就是"全列冻结 codes + 可训 `(s,z)`"。

## 9. 参数预算（Llama-2-7B，`group_k=128`）

| 项 | `group_size=32`（当前配置） | `group_size=64` |
|---|---|---|
| Salient 可训权重 | **157.3M** | 157.3M |
| 可训 `(s, z)` | 404.8M | 202.4M |
| 冻结 codes | 6318.7M（int8 6.3 GB/rank） | 同左 |
| 合计可训 | 562.0M | 359.7M |
| *对照* LoRA r=64（7 proj） | 159.9M | 159.9M |

**salient 段可训参数（157.3M）与 LoRA r=64（159.9M）几乎完全相等**——同样的训练预算，但在 salient 列子空间里
是满秩的，并且原生可部署。这个对照建议直接写进论文。`(s, z)` 看似额外，但**它们不是额外的部署开销**——GPTQ
checkpoint 本来就要存这些 metadata。

## 10. 成本账本（决定了默认配置）

设 `G_W = ∂L/∂W = gy^T x`，`N` = 目标线性层权重总数，`k` = salient 占比，`g` = group_size。
三组可训参数的代价差了两个数量级：

| 梯度 | 形式 | 代价 | 为什么 |
|---|---|---|---|
| `∂L/∂W_S` | `gy^T x_S` | **2N·k**（≈2%） | 只涉及 salient 那几列，一条窄条 GEMM |
| `∂L/∂z[o,g]` | `−s·Σ_t gy[t,o]·(Σ_{j∈g} x[t,j])` | **2N/g**（g=64 时 ≈1.5%） | 组内是**均匀求和**，可以先塌进 `x` 变成 pooled 输入，把 GEMM 的 K 维缩小 g 倍 |
| `∂L/∂s[o,g]` | `Σ_{j∈g}(q−z)[o,j]·G_W[o,j]` | **2N，不可约** | 逐元素因子 `(q−z)[o,j]` 同时依赖 `o` 和 `j`，塌不进 `x`（缺 o）也塌不进 `gy`（缺 j）。这是真正的三元缩并 `Σ_{t,j} gy[t,o]·x[t,j]·(q−z)[o,j]`，**无论怎么结合都要 T×out×in 次乘加，且与 s 的参数个数无关** |

所以总账是 `6N + 2N·k + 2N/g + 2N·[train_scale]`。**唯一昂贵的是 s。**

### 由此得到的默认配置：z-only

`train_scale: false` 是默认，不是妥协：

- 训练代价回到 `≈6N`，**与 PSQAT / QLoRA 同级**；
- 适配容量**严格大于** PSQAT 与 QA-LoRA 的并集——QA-LoRA 的 group-pooled adapter 恰好就是这个
  `Δz` 的 rank-r 受限参数化，这里是满秩的；
- **前向也一并变便宜**：s 冻结 ⇒ `q·s` 是常数，预计算一次存成 `wq`，于是
  ```
  y_N = x @ (q·s)ᵀ  −  pool_g(x) @ (z·s)ᵀ
  ```
  一个冻结 GEMM + 一个 `[T, in/g] × [in/g, out]` 的修正。**前向反向都不再出现 [out,in] 的重建张量**，
  普通 autograd 在这个写法上自动给出上表的廉价梯度，不需要自定义 `autograd.Function`。

> 早先版本这里写过"fp32 重建 + 独立 GEMM 是固有代价"——**那是错的**。可训练性不改变前向的形态，
> 额外代价全在反向；而反向的昂贵部分只来自 s。

`train_scale: true` 保留为消融，走 reconstruct-then-GEMM 的慢路径，明码标价 +2N。

## 11. 实现现状（已完成）

| 里程碑 | 状态 | 验收结果 |
|---|---|---|
| **M0** 重构抽出 `quant_primitives` / `permute_common` / `gptq` + re-export shim | ✅ | 4 个既有测试套件全绿，导入面无缺失 |
| **M1** `build_saltq_base`：GPTQ → 冻结 codes + `(s0,z0)` + salient 初值 | ✅ | e2e stage 2 |
| **M2** `SALTQLinear` + step-0 等价性 | ✅ | **asym 下 step-0 == GPTQ 重建，`max|Δ| == 0`** |
| **M3** 训练闭环：三组 optimizer、只存可训张量、DDP、resume | ✅ | e2e stage 4/6；checkpoint 体积 < 冻结基座 |
| **M4** `export_saltq`：无 merge 导出 + 等价性断言 | ✅ | **每层 deployed == trained，`max|Δ| == 0`** |

测试：`scripts/test_saltq.py`（40 项，严格 0 容差）+ `scripts/test_saltq_e2e.py`（17 项，tiny Llama 走完整链路）。

### 落地细节（踩过的坑）

- **`grad_scale` 的 1-ulp**：修成前向精确的 `autograd.Function`（见 §6 不变量 2）。这是让 `max|Δ| == 0`
  从 `< 1e-8` 变成真正 `== 0` 的唯一改动，且对既有 LSQ 路径只增不减。
- **codes 不进 state_dict**：`register_buffer(..., persistent=False)`，配合 `_SALTQTrainer._save` /
  `_load_from_checkpoint` 只读写可训张量。否则每个 checkpoint 会写 6.3 GB 不变的 int8。
- **DDP 不广播 codes**：`ddp_broadcast_buffers=False`（saltq 专用）。默认 True 会每步广播 6.3 GB。
- **`build_saltq_base` 的宿主内存**：`gptq_quantize_model_sequential` 返回的是 fp32 整数张量
  （7B ≈ 27 GB），转 int8 时用 `pop` 边转边释放，避免 fp32 与 int8 副本叠加。
- **dense 导出的 cast 是全链路唯一有损步骤**：`materialize_dense_model` 返回并打印该误差，
  绝不默认它是 0。真 packed 导出没有这一步。

## 12. 消融清单

0. **【第一优先】`z-only` vs `s+z`**（`train_scale`）。若 z-only 够用，§10 那 +2N 从根上消失，
   SALT-Q 与 PSQAT 同成本。先验预期 z 吃掉大部分收益：组内加性平移直接对抗量化的 clipping/偏置
   误差，而 QA-LoRA 仅靠它的**低秩**版本就能 work，本身就是证据。
1. **段拆分消融**：full SALT-Q / 只训 salient / 只训 z / 旧 Permuted SQAT。支撑"两半各自承接
   `BA` 的哪部分"的叙事。
2. `group_k` 扫描（1 / 2 / 4 组）与 `group_size` 扫描（32 / 64，直接改变 `(s,z)` 的自由度密度）。
3. `s` / `z` 分别冻结——验证 QA-LoRA "只训 z 且受 rank 限制"的退化关系。
4. `train_layernorms` on/off。
5. `salient_lr` 扫描：**首要超参**，训的是真实权重，LoRA 的 2e-4 会炸。默认 2e-5。
6. GPTQ 初始化 vs RTN 初始化——`(s,z)` 无法修正选错的 code，初始化质量在 2bit 下影响可能很大。
7. AWQ-S：当前 SALT-Q 未接入。先验判断是冗余（salient 权重自由 + LSQ 学 per-(行,组) scale，
   任何 per-channel 缩放都能被吸收）。若要做消融，`build_saltq_base` 里把 `awq_scales=None` 换掉即可。

## 13. 后续（未做）

- ~~P2 自定义 `autograd.Function`~~：**z-only 下已不需要**——pooled-GEMM 写法让普通 autograd
  直接给出廉价梯度，且全程不物化 `W_eff`。只有 `train_scale=true` 的消融路径还会重建权重。
- **codes 位打包**：当前 int8 6.3 GB/rank。46 GB 卡够用；换 24 GB 卡或更大模型时必须打包
  （2bit→1.6 GB、3bit→2.4 GB），`SALTQLinear.codes` 是唯一需要改的接口。
- **真 packed GPTQ checkpoint 导出**：现有 `export.pack_int4` 是 4bit + AWQ 列序 + `AWQ_ZERO_POINT=8`
  的对称约定，2/3bit 需要新 packer。届时 §11 里那个 dense cast 误差归零。
- **code refresh**：codes 冻结是方法卖点，但若实验显示是瓶颈，可在训练中期用当前 `(s,z)` 与一个 CPU 上的
  fp32 shadow 重新取整一次。
- **融合**：同 block 内 q/k/v、gate/up 的重建 + GEMM 合并（沿用 `_FusedSiblingQATInjector` 思路）。

## 14. 怎么跑

```bash
bash runs/saltq/run_saltq_math.sh                      # train + export + eval（MetaMathQA, INT3 g64, 3 GPU）
bash runs/saltq/run_saltq_math.sh --bits 2             # INT2 g64（run3）
bash runs/saltq/run_saltq_math.sh --salient_lr 1e-5    # 扫首要超参
bash runs/saltq/run_saltq_math.sh --skip_train --checkpoint_dir outputs/saltq-2bit-saltq/final
bash runs/saltq/run_saltq_math.sh --resume_from outputs/saltq-2bit-saltq/checkpoint-500
```

两个离线基座（`outputs/saltq/permuted_fp16_base` ~13 GB、`outputs/saltq/saltq_base*` ~7 GB/bit）在
resume 和 export 时**必须复用**，不要手工删除。

结果表：`results_saltq.csv`（长表）。reported baseline 来自 `baselines_metamath.csv`（`source=report`），
本机跑出来的走 lm-eval（`source=lm-eval`）——**这一列必须保持诚实**，不要把没测过的数字标成 lm-eval。

### run 记录

| run | 配置 | GSM8k (strict) | 结论 |
|---|---|---|---|
| run1 | INT3 g64，**整数 z + scale 量级 lr** | 32.9 | **作废**：z 完全没动（round(z) 在 1.63M 个参数上零变化），非显著段贡献为 0 |
| run2 | INT3 g64，连续 z，z-only，salient_lr 2e-4 / zp_lr 1e-3 | **44.8** | 超过 PSQAT+GPTQ 42.6 **+2.2**，逼平 LR-QAT/fp16 45.2 |
| run3 | **INT2 g64**，其余与 run2 逐字一致 | 31.8 | **lr 没跟着比特走**：INT2 的格步长是 INT3 的 7/3 倍，同一个 lr 只把 W_S 推了 0.25 个格步（run2 是 0.58）——见下 |
| run4 | **INT2 g32**，salient_lr 5e-4 / scales_lr 2e-5 / zp_lr 1e-3，eff batch 80 | **36.1** | 比 run3 +4.3，但**同时动了两个变量**（group_size 64→32 与 lr 修正），不可归因。缺 INT2 g64 @ 5e-4 这个点来解耦 |
| zonly | 与 run4 逐字一致，只翻 `train_salient: false` | 29.8 | **干净的单变量消融**：砍掉 salient 权重训练掉 **6.3 分**。1–2% 的权重扛了 6.3 分，说明 INT2 下高精度 latent + LSQ 这一档贡献很大 |
| zp5e3 | 与 run4 逐字一致，只把 `zp_lr_by_bits[2]` 1e-3→5e-3 | **40.9** | **+4.8**。依据是 run4 的实测位移（见下）：z 是唯一欠驱动的一档。**INT2 g32 当前最好**（flexible 37.3，低于 strict，是 flexible 抽取误取数字所致，不影响 strict 口径） |

### lr 必须跟着比特走（run3 的教训，已量化）

用 `scripts/measure_saltq_displacement.py` 量出来的实测（p50）：

| | 格步长 s | \|ΔW_S\| (格步) | \|Δs_S\| 相对 | \|Δz_N\| (level) | c=位移/lr |
|---|---|---|---|---|---|
| run2 INT3 lr 对 | 1.22e-2 | **0.584** | 3.3% | 0.039 | 35.5 / 40.1 / 38.8 |
| run3 INT2 lr 照抄 | 2.84e-2 | **0.246** | 1.4% | 0.040 | 34.9 / 40.7 / 40.4 |
| run4 INT2 g32 lr 修正 | 2.49e-2 | **0.504** | 2.8% | **0.032** | 25.1 / — / 32 |

run4 追加的两条结论：

- **weight 单位的两档已经到位，z 没有**。0.504 格步、2.79% 相对——都在靶心；而 z 只走了 0.032 level，
  对着 0.1–0.3 的有用区间**差 3–9 倍**，比 run2 的 0.039 还低。z 管 98–99% 的权重，且 INT2 正是
  non-salient 段靠 GPTQ 压不住的失效区。**再抬 salient_lr 没有收益**——0.584 是 run2 拿到 44.8 的
  已验证工作点，越过它是新赌注不是补自由度。
- **`c` 不是常数，是 ∝ √T，而 T 取决于 eff batch**。run4 从 3 卡换到 4 卡，eff batch 60→80，
  T 6584→约 4938，实测 c 从 35.5 掉到 **25.1**。5e-4 之所以还能命中 0.5 格步，是因为 g32 的格步长
  p50（2.49e-2）比推导用的 g64 值（2.84e-2）小 12%，两个偏差恰好抵消——**别指望这种运气**。
  改 epochs 或 eff batch 时，所有 lr 要乘 √(T_new/T_old)。要在 4 卡上拿回 eff batch 60：
  `per_device 3 × accum 5 × 4`。

三件事被前两组数钉死了：

1. **Adam 的位移只由 lr 决定**：三个参数组、两个比特宽度下，`c = 位移/lr` 全部落在 35–41
   （≈0.47·√T，T=6584）。梯度幅值、LSQ 的 1/√(N·Qp) 梯度缩放在 Adam 里被 v̂ 归一化整除掉了
   ——**所以 lr 必须按参数自身的单位来定，这就是按单位分组的全部理由**。（`c` 随 T 变，见上。）
2. **min-max 初始化让 s(b) = range/(2^b−1) 精确成立**：实测 s₂/s₃ = 2.333 = 7/3（四位有效数字），
   salient / non-salient 都是。所以 weight 单位的两个 lr 必须 ∝ 1/(2^b−1)。
3. **run3 的位移正好是 run2 的 3/7**：0.584×3/7 = 0.250，实测 0.246。它不是"2bit 学不动"，
   是**只给了一半的自由度**。s_S 同理（3.3%×3/7 = 1.4%，实测 1.43%）。

结论表已写进 `configs/saltq.yaml`（`*_by_bits`，由 `src/trainer.py:saltq_lr()` 按 `quant_bits`
解析，命令行显式 lr 仍然最高优先）：

| bits | salient_lr | scales_lr | zp_lr |
|---|---|---|---|
| 2 | 5.0e-4 | 2.0e-5 | 1.0e-3 |
| 3 | **2.0e-4**（已验证） | **1.0e-5** | **1.0e-3** |
| 4 | 1.0e-4 | 5.0e-6 | 2.0e-3 |

zp_lr 不随比特下降，因为 z 的单位是 level：它一半的活是补 ±0.5 level 的量化误差（与比特无关），
另一半是把固定 weight 量级的任务 delta 换算成 level（∝ 2^b−1，所以只在 INT4 抬到 2e-3）。

**已验证：zp_lr 是当时唯一欠驱动的一档，抬它值 +4.8 分。** run4（INT2 g32）实测 z 只走了 **0.032**
level，有用区间 0.1–0.3——只吃掉约 6% 的码字误差；而 weight 单位的两档（0.504 格步 / 2.79%）都已在靶心。
早先写的 `zp_lr=3e-3` 是从 INT3 的 0.039 外推的，INT2 实测更低（`c_z = 0.032/1e-3 = 32`），3e-3 只到
~0.096 **仍在区间外**，所以直接跳到 **5e-3**（→ ~0.16，区间中点）。结果 **36.1 → 40.9**
（`configs/saltq_zp5e3.yaml`，run `20260809_162312`）。

这条给出的方法论比这一个数字更重要：**先用 `measure_saltq_displacement.py` 量位移，再决定动哪个 lr。**
run1/run3 两次作废、以及"该不该继续抬 salient_lr"这个问题，都是靠这把尺子在零 GPU 成本下解掉的。

下一步的余量：5e-3 是否已经到位仍未量（应重跑一次 displacement 确认 z 落在 0.1–0.3 而不是冲过 0.5）；
INT2 下 z 的值域只有 `[0, 3]`，0.16 level 约占 5%，clamp 风险小，但 7e-3/1e-2 是否继续涨要看实测。

run2 的三点对照（这一格是唯一有完整外部 baseline 的）：

| 参照 | GSM8k | 对 SALT-Q 的含义 |
|---|---|---|
| LR-QAT / fp16 | 45.2 | 精度天花板，成本 +175% |
| **PSQAT+GPTQ** | **42.6** | **必须打过的下限**（同量级成本）→ 已打过 |
| SalientFP16+GPTQ | 42.3 | 说明高精度通道不是必需品 |

### 跑 bit 扫描时的基座纪律

- `permuted_fp16_base` **与比特无关**（它在任何量化之前就建好了），所以 3bit→2bit **必须复用同一份**：
  重建会重跑校准、可能给出不同的分段，那样两个 run 就不止差一个变量了。`scripts/train.py` 里
  `_permuted_base_reusable()` 会在分段配置一致时自动复用。
- `saltq_base`（冻结 codes）**与比特强相关**。默认路径已改成
  `<output_dir>/saltq_base_{bits}bit_g{gs}`（旧的无后缀目录在配置匹配时仍然复用，所以 3bit 的
  checkpoint 指针不会失效）。若某个路径下已有一份 **配置不符** 的 base，训练会直接报错而不是覆盖
  ——codes 是一次性的离散选择，覆盖它会让指向它的 checkpoint 永久无法 export。

**不要用 loss 值判断"量化有没有生效"。** 我最初写的是"初始 loss 应落在 1.8–2.5，≈1.0 则可疑"，
实测首个 INT3 g64 run 是 **1.219**，一度看着像量化没生效。**是这个阈值错了，不是训练错了**：

- PSQAT 报告的 step-0 loss 1.8 是用 **min-max RTN 砸 salient 列**得到的——而那恰恰是 min-max 最
  吃亏的地方（幅值最大的 outlier 列）；
- SALT-Q 的 step-0 是**整个权重都经过 GPTQ 误差补偿**的模型，GPTQ 最小化的是输出误差而非权重误差。
  **量化的列更多，起点反而更好**，所以 loss 低于 PSQAT 是预期结果。

唯一可靠的判据是直接查基座，`python scripts/check_saltq_base.py`：

- **code 直方图**：INT-b affine 必须铺满 `[0, 2^b−1]`；高斯权重下应是钟形（实测 INT3
  `[4,8,16,23,23,16,8,4]%`）。塌到 ≤2 个格点 = 网格坏了。
- **重建误差**：`(q−z)·s` 对 permuted fp16 权重的相对 Frobenius 误差。INT3 就该是**几十个百分点**
  （实测 24–40%）。接近 0 才说明"量化后的权重"其实还是全精度。
- **group_k 结构**：o_proj 必须为 0，其余为 group_size 的正整数倍。

loss ≈ 10 仍然是"立刻停"的信号（历史上 LR-QAT 那个 loss=10 正是实现 bug 的表现，不是低比特的
固有现象），但下限不设阈值。
