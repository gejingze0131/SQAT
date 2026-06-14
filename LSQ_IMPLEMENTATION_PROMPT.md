# 任务:为 full(LR-QAT) 与 sqat_permute(salient slice) 引入 LSQ 式可学习量化 scale(可选开关 enable_lsq)

## 0. 背景与目标
本仓库(`/home/jingze/ssd/projects/SQAT`)的 QAT 量化目前用**每步动态 min-max scale**(权重组 `amax/q_max`,asym 用 pos/neg affine)。低 bit(W3g64)下 min-max 对 outlier 敏感、scale 随权重每步跳变 → full QAT 初始 loss≈10(=随机, `ln(32000)≈10.4`)、训练不稳。

LR-QAT(arXiv:2406.06385, github `Qualcomm-AI-research/LR-QAT`)用 **LSQ 式可学习 scale**(`--qmethod symmetric_uniform` 但 **W2 用 asymmetric**, `--learn-ranges`, `--scales-lr 1e-5`, `current_minmax` 初始化)在 W2/W3 仍稳定。

目标:把"可学习量化 scale(LSQ / asym 时 LSQ+)"作为**可选开关 `enable_lsq`** 引入 full 与 sqat_permute,**完全保留原 min-max 实现**(enable_lsq=false 时逐 bit 等于现状)。

## 1. 硬约束(本仓库踩过的坑,务必遵守)
1. **训练 fakequant 的网格必须与导出 PTQ 网格逐 bit 一致**。历史 bug:训练 min/max、导出 pos/neg,导致 QAT 完全失效。LSQ 引入后**训练用学到的 scale,导出必须用同一个学到的 scale**(不能重新 min-max)。**必须写数值测试**:`train_lsq_fakequant(W, s*[,z*]) == export_dequant(export_quant(W, s*[,z*]))`,max|Δ|<1e-5。
2. 所有改动用 `enable_lsq` 门控,默认 false。
3. Python:`/home/jingze/miniconda3/envs/sqat/bin/python`(torch+bnb+CUDA)。CPU 单测加 `CUDA_VISIBLE_DEVICES=""`;真实 bnb-4bit 测试需 GPU(先 `nvidia-smi` 选空闲卡,如卡 1)。临时脚本用 `scripts/_tmp_*.py` 命名,用完删除。
4. 现有测试必须全过:`scripts/test_qat_permute_sqat.py`(21 项)、`scripts/test_gptq_nonsalient.py`。
5. 先读这些文件再动手:
   - `src/qat_base.py`(`round_ste`, `groupwise_symmetric_fakequant`, `groupwise_asymmetric_fakequant`, `asymmetric_scale_zero_from_pos_neg`, `FullQATLoRAInjector`, `FullQAT`)
   - `src/qat_permute_sqat.py`(`round_ste`, `_asym_qparams`, `_sym_scale`, `group_fakequant`, `group_quantize`, `group_dequantize`, `verify_permute_quant_consistency`, `fused_qat_residual_outputs`, `_FusedSiblingQATInjector`/`FusedAttnQATInjector`/`FusedMLPQATInjector`/`DownProjQATInjector`, `install_fused_selective_qat`, `gptq_quantize_layer`, `gptq_quantize_model_sequential`, `awq_s_for_module`, `compute_awq_scales`, `build_permuted_fp16_checkpoint`, `SegmentPermutedSelectiveQAT.prepare_model`)
   - `src/export.py`(`real_quantize_symmetric/asymmetric`, `real_quantize_fixed_symmetric/asymmetric` line~144/161, `dequantize_symmetric/asymmetric`, `_quantize_all_layers`, `merge_and_export`, `_verify_ptq_consistency`, `save_sqat_permute_meta`, `collect_sqat_permute_metadata`)
   - `scripts/train.py`(`load_config`, `get_qat_handler`, `build_trainer` 调用, sqat_permute 的 build_permuted 预步骤, export-only 分支)
   - `src/trainer.py`(`QATCallback`, optimizer 创建)
   - `configs/sqat_permute_math.yaml`, `configs/sqat_permute_commonsense.yaml`
   - `run_full_qlora.sh`, `run_permute_sqat.sh`

## 2. LSQ / LSQ+ 规范代码(钉死,直接用)
```python
def grad_scale(x, g):
    return (x - x * g).detach() + x * g            # forward=x; backward: grad *= g
# round_pass == 现有 round_ste

# ---- symmetric LSQ(只学 scale)----
def groupwise_lsq_symmetric_fakequant(W, scale, group_size, q_bits, eps=1e-8):
    # W:[out,in] fp32 ; scale:[out,ng] (>0, learnable Parameter)
    Qn, Qp = -2**(q_bits-1), 2**(q_bits-1)-1
    Wg = _group_reshape(W, group_size)                         # [out,ng,gs] (pad in_f→ng*gs)
    s  = grad_scale(scale.clamp(min=eps)[..., None], 1.0/math.sqrt(group_size*max(Qp,1)))
    Wq = round_ste((Wg / s).clamp(Qn, Qp)) * s
    return _ungroup(Wq, in_f)                                   # [out,in]

# ---- asymmetric LSQ+(学 scale + 学 zero-point;默认配置走这条)----
def groupwise_lsq_asym_fakequant(W, scale, zp, group_size, q_bits, eps=1e-8):
    # scale:[out,ng] (>0, learnable) ; zp:[out,ng] (learnable real)
    Qn, Qp = 0, 2**q_bits - 1
    Wg = _group_reshape(W, group_size)
    s  = grad_scale(scale.clamp(min=eps)[..., None], 1.0/math.sqrt(group_size*max(Qp,1)))
    z  = grad_scale(zp[..., None],                    1.0/math.sqrt(group_size))
    z  = round_ste(z).clamp(Qn, Qp)                            # zero-point 训练即取整(STE)
    q  = round_ste((Wg / s + z).clamp(Qn, Qp))
    Wq = (q - z) * s                                           # dequant = (q - z)*s
    return _ungroup(Wq, in_f)

# ---- 初始化(current_minmax,保证 enable_lsq 初始网格 == 原 min-max,无跳变)----
@torch.no_grad()
def init_lsq_scale_sym(W, group_size, q_bits):                 # -> scale[out,ng]
    Qp = 2**(q_bits-1)-1
    return (_group_reshape(W, group_size).abs().amax(-1) / Qp).clamp(min=1e-8)
@torch.no_grad()
def init_lsq_scale_zp_asym(W, group_size, q_bits):             # -> (scale[out,ng], zp[out,ng])
    # 复用现有 _asym_qparams 的 min/max 公式: scale=(wmax-wmin)/qmax, zp=round(-wmin/scale)
    ...

# ---- 导出:用学到的 scale[,zp] 固定量化(与训练同一套 Qn/Qp)----
@torch.no_grad()
def lsq_quantize_export_sym(W, scale, group_size, q_bits):     # -> W_int[out,in]
    Qn,Qp = -2**(q_bits-1), 2**(q_bits-1)-1
    return _ungroup((_group_reshape(W,group_size)/scale[...,None]).round().clamp(Qn,Qp), in_f)
@torch.no_grad()
def lsq_quantize_export_asym(W, scale, zp, group_size, q_bits):# -> (W_int, z_int)
    Qn,Qp = 0, 2**q_bits-1
    z_int = zp.round().clamp(Qn,Qp)                            # 训练 round_ste、导出 round → 同值
    q = ((_group_reshape(W,group_size)/scale[...,None]) + z_int[...,None]).round().clamp(Qn,Qp)
    return _ungroup(q, in_f), z_int                           # dequant = (q - z_int)*scale
```
**关键约定**:LSQ 自成一套 single-source-of-truth;sym 用 `Qn=-2^(b-1), Qp=2^(b-1)-1`(比现有 `[-q_max,q_max]` 多一个负 level),asym 用 `Qn=0,Qp=2^b-1`。**不要复用现有 symmetric 的 `[-q_max,q_max]`**(Qn 约定不同会破坏一致性)。

## 3. 设计决策(照此实现)
1. **enable_lsq 跟随 `cfg["qat"]["symmetric"]`**:asym → LSQ+(学 scale+zp,zp 训练即 `round_ste`);sym → 经典 LSQ(只学 scale)。默认配置是 asym → 默认 LSQ+。初始化一律 current_minmax。
2. **scale[,zp] 是 model tree 内的 `nn.Parameter`**(让 HF Trainer optimizer 能收录),`requires_grad=True`,**在 build_trainer(optimizer 创建)之前**(prepare_model 阶段)注册:
   - full:每个 target PEFT module 注册 `module.lsq_w_scale`(asym 再加 `module.lsq_w_zp`),shape `[out, in//group_size]`,init 用 `dequant(base)`(LoRA B 初始为 0,W_curr≈base)。
   - sqat_permute(salient slice):每个注入器对应融合切片 `[sum_out, group_k//group_size]`。注册到 **model tree 内的稳定 module**(建议 `self_attn`/`mlp` block,或对应的 `q_proj`/`down_proj`),参数名含 `lsq_w_scale`(asym 加 `lsq_w_zp`),init 从 `W_base_salient` 算。
3. **单独 LR + 单独存盘**:
   - `src/trainer.py` optimizer 创建处:把名字含 `lsq_w_scale`/`lsq_w_zp` 的参数单独成一个 param group,`lr=cfg["qat"]["lsq"]["scales_lr"]`(默认 1e-5),`weight_decay=0`;其余参数组不变。**打印该组参数数量确认被优化**(>0)。
   - scale/zp 不会被 PEFT `save_pretrained` 保存 → 在 `on_train_end`/collect 时存:full 存到 checkpoint dir 的 `lsq_scales.pt`(`{module_name: {"scale":..., "zp":...}}`);sqat_permute 并入 `model._sqat_permute_meta["lsq_scales"]`(`{layer_proj_key: {"scale":..., "zp":...}}`)+ `_sqat_permute_meta["lsq"]=True`。
4. **导出一致性(最关键)**:
   - full:`_quantize_all_layers` 在 enable_lsq+`qat_mode=="full"` 时,用 `lsq_quantize_export_*` + dequant=`(q[-z])*scale`(读 `lsq_scales.pt`),不要 min-max。
   - sqat_permute:salient slice 的 `group_quantize` 与 `gptq_quantize_layer` 的 `[0:group_k]` 固定网格改用学到的 scale[,zp]。给 `group_quantize` 加可选 `fixed_scale=(scale[,zp])`;`gptq_quantize_layer` 的 salient 固定网格用 `fixed_scale`。**非 salient 的 GPTQ/min-max 路径不变**。
   - **测试**:对随机 `W` 和随机正 `scale`(+ 随机 `zp`),断言 `groupwise_lsq_*_fakequant(W,s[,z]) == dequant(lsq_quantize_export_*(W,s[,z]))`,max|Δ|<1e-5,覆盖 sym/asym × INT3/INT4 × g64/g128。
5. **LSQ 与 awq_scale 可共存(非互斥)**:AWQ 的 `*S … /S` 是 fakequant **外层**(按输入列),LSQ 改的是 **内层** scale。叠加只需把 `fused_qat_residual_outputs` 里 `awq_s` 分支内层的 `group_fakequant` 换成 LSQ 版(scale/zp 学在 amplified 空间),**外层 `*S//S` 与 GPTQ 非 salient 路径一行不动**;导出量化 `W_merged_S*S` 用学到的 scale/zp,dequant 后 `/S` bake-back。

## 4. 改动清单(逐文件逐函数)
### src/qat_base.py
- 加 `grad_scale`, `groupwise_lsq_symmetric_fakequant`, `groupwise_lsq_asym_fakequant`, `init_lsq_scale_sym`, `init_lsq_scale_zp_asym`, `lsq_quantize_export_sym`, `lsq_quantize_export_asym`(+ `_group_reshape`/`_ungroup` 小工具,或复用现有 reshape 逻辑)。
- `FullQATLoRAInjector.__init__` 加 `enable_lsq`;true 时:按 cfg.symmetric 注册 `m.lsq_w_scale`(asym 再注册 `m.lsq_w_zp`),init 从 `dequant(base)`;`_forward` 用 LSQ fakequant(读 `m.lsq_w_scale[, m.lsq_w_zp]`)替代 min-max。false 时走现有路径。
- `FullQAT.prepare_model` 读 `cfg["qat"].get("lsq",{}).get("enabled",False)` + symmetric,传给 injector;收集 scale/zp 引用。`FullQAT.on_train_end` 存 `lsq_scales.pt`。

### src/qat_permute_sqat.py
- `fused_qat_residual_outputs` 加可选 `lsq_scale`/`lsq_zp`;给出时内层用 LSQ 版(symmetric/asym 跟随);与 `awq_s` 叠加时:`W_use = W_curr*awq_s` → LSQ fakequant → `/awq_s`。
- `_FusedSiblingQATInjector`/`DownProjQATInjector` 加 `enable_lsq`+`symmetric`:注册 `lsq_w_scale`(+`lsq_w_zp`)到 model tree(block 或 proj),init 从 `W_base_salient`,`_pre_hook`/`_hook` 传给 `fused_qat_residual_outputs`。
- `install_fused_selective_qat`/`prepare_model` 传 `enable_lsq`;把 scale/zp 收进 `model._sqat_permute_meta["lsq_scales"]` 与 `["lsq"]=True`。
- `group_quantize` 加可选 `fixed_scale`(sym: `scale`;asym: `(scale,zp)`;给出时跳过 amax/_asym_qparams,直接用)。`gptq_quantize_layer` salient 固定网格支持 `fixed_scale`。`verify_permute_quant_consistency` 支持 LSQ 网格校验。

### src/export.py
- `_quantize_all_layers` 加参数 `lsq_scales`/`enable_lsq`;full 分支用 `lsq_quantize_export_*`;sqat_permute 分支给 `group_quantize`/`gptq` 传 `fixed_scale`(从 meta `lsq_scales` 取,asym 含 zp;若 awq 同开则先 `*S`)。
- dequant save loop:full 用 `(q[-z])*scale`;sqat_permute 用 `group_dequantize`(支持 fixed scale/zp)+ 若 awq 则 `/S` bake-back。
- `merge_and_export`:enable_lsq 解析(full 从 `lsq_scales.pt`,sqat_permute 从 meta);校验打印 train↔export LSQ 网格一致。

### src/trainer.py
- optimizer 创建:`lsq_w_scale`/`lsq_w_zp` 单独 param group(lr=`scales_lr`, wd=0),打印数量。

### scripts/train.py
- `--enable_lsq`/`--no_enable_lsq` 三态 → `cfg["qat"]["lsq"]["enabled"]`;export-only:full 从 `lsq_scales.pt`、sqat_permute 从 meta 读 enable+scale。

### configs/*.yaml(两个)
```yaml
qat:
  lsq:
    enabled: false
    scales_lr: 1.0e-5
```

### run_full_qlora.sh / run_permute_sqat.sh
- 加 `ENABLE_LSQ` 开关 → `--enable_lsq/--no_enable_lsq`(train 与 export-only 命令都带)。

## 5. 分三阶段交付
- **Phase 1(先做并 gating)**:仅 full QAT 的 LSQ(qat_base.py + trainer optimizer group + export full path + 数值一致性测试)。用 `bash run_full_qlora.sh`(W3g64,asym→LSQ+)验证 **初始 loss 从合理值开始(远小于 10)、训练稳定**。这是判断 LSQ 是否解决问题的关键信号;不达标先不进 Phase 2。
- **Phase 2**:sqat_permute salient slice 的 LSQ(qat_permute_sqat.py + export salient/gptq `fixed_scale` + 测试),awq 关。
- **Phase 3(可选)**:叠加 awq+LSQ,并加叠加态的 train==export 测试。

## 6. 验证清单(每阶段必做)
1. `enable_lsq=false` 回归:full 与 sqat_permute 的训练 forward / 导出结果与改动前逐 bit 一致。
2. LSQ train fakequant == LSQ export quant→dequant(同 scale[,zp]),max|Δ|<1e-5(sym/asym × INT3/4 × g64/g128)。
3. scale[,zp] 进入 optimizer、`requires_grad`、训练中梯度非零且值变化。
4. 真实 bnb-4bit GPU 小测(full):LSQ injector forward 正确、base 冻结、梯度到 LoRA+scale、`checkpoint(use_reentrant=False)` 兼容、remove()/导出干净。
5. `scripts/test_qat_permute_sqat.py` 与 `scripts/test_gptq_nonsalient.py` 全过。

## 7. 注意/风险
- HF Trainer 通常收录所有 `requires_grad` 参数,但 PEFT 过滤可能漏掉自注册参数 → **务必在 trainer.py 显式建 param group 并打印确认**。
- scale 必须 >0(`clamp(min=eps)`);发散先调小 `scales_lr` 或设 0(冻结 scale)对照。
- 注册 scale 必须在 optimizer 创建之前(prepare_model 阶段)。
- gradient checkpointing(use_reentrant=False)下 LSQ scale 二次 forward 的梯度需正确累加(测试覆盖)。
- asym zero-point 训练用 `round_ste`、导出用 `round` → 同值,是 train/export 一致的关键,勿改成实数 offset。

## Sources
- LSQ: arXiv:1902.08153 ; LSQ+: arXiv:2004.09576 ; LR-QAT: arXiv:2406.06385(repo: Qualcomm-AI-research/LR-QAT)
- 规范 LSQ 实现参考: github hustzxd/LSQuantization/lsq.py
