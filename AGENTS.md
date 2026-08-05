# AGENTS.md — SQAT 仓库工作指南 & SALT-Q

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

- QAT 模式字符串：`"saltq"`　主模块：`src/qat_saltq.py`　配置：`configs/saltq.yaml`　启动：`run_saltq.sh`
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

## 10. 前向 / 反向

```
W_S_eff = LSQ_fakequant(W_S ; s_S, z_S)          # W_S / s_S / z_S 可训，STE
W_N_eff = (q_N − round_ste(z_N)) · s_N           # q_N 冻结 int8，s_N / z_N 可训
y = x @ cat([W_S_eff, W_N_eff], dim=1)ᵀ
```

关于"不 materialize"的准确表述：对 `(s, z)` 的梯度确实可以闭式约简
（`∂L/∂s[o,g] = Σ_j G_W·(q−z)`、`∂L/∂z[o,g] = −s·Σ_j G_W`），但 **`∂L/∂x = gy @ W_eff` 绕不开
`[out, in]` 规模的重建**。当前实现（P1）用普通 autograd 每次 forward 重建 `W_eff`，开销 profile 与既有
`FullQATLoRAInjector` 一致，梯度检查点把它限制在一层之内。P2（自定义 `autograd.Function`，forward 不保存
`W_eff`）见 §13。

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

1. **段拆分消融**：full SALT-Q / 只训 salient / 只训 `(s,z)` / 旧 Permuted SQAT。支撑"两半各自承接
   `BA` 的哪部分"的叙事。
2. `group_k` 扫描（1 / 2 / 4 组）与 `group_size` 扫描（32 / 64，直接改变 `(s,z)` 的自由度密度）。
3. `s` / `z` 分别冻结——验证 QA-LoRA "只训 z 且受 rank 限制"的退化关系。
4. `train_layernorms` on/off。
5. `salient_lr` 扫描：**首要超参**，训的是真实权重，LoRA 的 2e-4 会炸。默认 2e-5。
6. GPTQ 初始化 vs RTN 初始化——`(s,z)` 无法修正选错的 code，初始化质量在 2bit 下影响可能很大。
7. AWQ-S：当前 SALT-Q 未接入。先验判断是冗余（salient 权重自由 + LSQ 学 per-(行,组) scale，
   任何 per-channel 缩放都能被吸收）。若要做消融，`build_saltq_base` 里把 `awq_scales=None` 换掉即可。

## 13. 后续（未做）

- **P2 自定义 `autograd.Function`**：forward 不保存 `W_eff`，backward 重建并用约简式求 `∂L/∂s, ∂L/∂z`。
- **codes 位打包**：当前 int8 6.3 GB/rank。46 GB 卡够用；换 24 GB 卡或更大模型时必须打包
  （2bit→1.6 GB、3bit→2.4 GB），`SALTQLinear.codes` 是唯一需要改的接口。
- **真 packed GPTQ checkpoint 导出**：现有 `export.pack_int4` 是 4bit + AWQ 列序 + `AWQ_ZERO_POINT=8`
  的对称约定，2/3bit 需要新 packer。届时 §11 里那个 dense cast 误差归零。
- **code refresh**：codes 冻结是方法卖点，但若实验显示是瓶颈，可在训练中期用当前 `(s,z)` 与一个 CPU 上的
  fp32 shadow 重新取整一次。
- **融合**：同 block 内 q/k/v、gate/up 的重建 + GEMM 合并（沿用 `_FusedSiblingQATInjector` 思路）。

## 14. 怎么跑

```bash
bash run_saltq.sh                      # train + export + eval（MetaMathQA, INT2, 3 GPU）
bash run_saltq.sh --salient_lr 1e-5    # 扫首要超参
bash run_saltq.sh --skip_train --checkpoint_dir outputs/saltq-2bit-saltq/final
bash run_saltq.sh --resume_from outputs/saltq-2bit-saltq/checkpoint-500
```

两个离线基座（`outputs/saltq/permuted_fp16_base` ~13 GB、`outputs/saltq/saltq_base` ~7 GB）在 resume 和
export 时**必须复用**，不要手工删除。

结果表：`results_saltq.csv`（长表）。reported baseline 来自 `baselines_metamath.csv`（`source=report`），
本机跑出来的走 lm-eval（`source=lm-eval`）——**这一列必须保持诚实**，不要把没测过的数字标成 lm-eval。

### 首个 run 的预期与诊断

第一个 run 是 **INT3 g64**，这一格有完整的三点对照：

| 参照 | GSM8k | 对 SALT-Q 的含义 |
|---|---|---|
| LR-QAT / fp16 | 45.2 | 精度天花板，成本 +175% |
| **PSQAT+GPTQ** | **42.6** | **必须打过的下限**（同量级成本） |
| SalientFP16+GPTQ | 42.3 | 说明高精度通道不是必需品 |

要证明的是"低成本吃掉那 2.6 分缺口"，所以 **42.6 是及格线，45.2 才是目标**。

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
