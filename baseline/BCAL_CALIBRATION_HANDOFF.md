# 给 LoTA-QAF / QWHA 复现 Agent:在 bcal(balanced-calibration)格子里构建 GPTQ 基座并提交

## 1. 背景:为什么要重建基座

主表里所有 GPTQ 基座此前都用训练文件的**前 N 条记录**校准(`calibration_sampling=first`)。
`datasets/commonsense` 的训练文件按任务分组,所以那 N 条 = **100% BoolQ、约 9.5k 有效 token**。
同一个 QLoRA merged checkpoint 只换校准集(`results_saltq.csv`,INT2 g32 span):

| 校准 | 有效 token | MEAN(8) |
|---|---|---|
| BoolQ-only 128 条(旧) | 9.5k | 36.64 |
| balanced 128 条 | 17k | 59.33 |
| BoolQ-only 3500 条 | 256k | 44.93 |
| **balanced 3500 条(D,新标准)** | **471k** | **66.22** |
| C4 128×2048(GPTQ 论文标准) | 262k | 30.67(任务模板丢失) |

任务混合是主因,预算其次;通用语料(C4/WikiText)在 INT2 会丢掉指令模板,所以**校准集必须是 in-domain 且任务均衡**。
INT3 floor 同样从 72.60 → 76.05。因此现在的主格子是 **bcal**:所有方法的基座都用 balanced 3500 条校准。
你们的行必须建在同样的校准上,否则与 bcal 格子不可比。

bcal 格子当前参照(Commonsense-170k,1 epoch,`loss_span=instruction+response`,MEAN(8),vLLM greedy):

| | INT3 g64 | INT2 g32 |
|---|---|---|
| QLoRA merged fp16(上限) | 77.75 | 77.75 |
| GPTQ floor(merged → GPTQ,balanced 3500) | 76.05 | 66.22 |
| QA-LoRA | 77.71 | 72.96 |
| SALT-Q | 78.85 | 74.42 |

## 2. 我们的校准机制(所有基座共用)

`src/data.py::load_calibration_data(cfg, tokenizer)` 读 `cfg["qat"]["sqat"]`:

```yaml
qat:
  sqat:
    calibration_samples: 3500        # 读入的训练记录数
    calibration_sampling: balanced   # first | shuffle | balanced;balanced = 每个 `type` 取 3500/8 条,seed 0
    calibration_seq_len: 2048        # 截断上限(记录本身 ~130 token,不拼接、不开窗)
    # calibration_source: datasets/c4_calib_1024.json   # 只在做通用语料对照时用,不要用于主行
```

返回的是**带 prompt 模板的完整记录**(tokenize 方式与训练完全相同),padding 位置在 Hessian 里被 mask 掉
(`src/gptq.py::_masked_xtx`)。`src/gptq.py::gptq_quantize_model_sequential(..., nsamples=N)` 只取 loader 里
前 N 条序列,所以 **GPTQ 的 `nsamples` 必须与 `calibration_samples` 相等(3500)**,否则只用了前几百条。
GPTQ 本身:静态分组(static groups)、无 act-order、percdamp 0.01、blocksize 128;3500 条在一张 GPU 上约 45 min
(catcher 会把 3500 条的层输入缓存在 CPU 内存里,几 GB,正常)。

可以直接复制的例子:`configs/qalora_cs170k_int2_g32_ep1_span_bcal.yaml`、`configs/saltq_cs170k_int2_g32_ep1_span_k256_bcal_sgptq.yaml`、
`jobs/cs_qlora_int2_span_floor_calibD.pbs`(D floor:`scripts/export_gptq_dequant.py --nsamples 3500 --calibration_samples 3500 --calibration_sampling balanced --batch_size 8`)。

## 3. LoTA-QAF:用 matched base 路线

`baseline/LoTA-QAF/sqat/make_matched_base.py` 已经走我们的 GPTQ + 我们的 `load_calibration_data`,再打包成 GPTQModel 格式
(打包不重新量化,codes 与我们的网格逐位相同,脚本自带回读断言)。它读两处配置:

- 数据:`qat.sqat.{calibration_samples, calibration_sampling, calibration_seq_len}`
- sweep:`qat.sqat_permute.gptq.{nsamples, batch_size, percdamp, blocksize}`

**注意**:`configs/qalora_*_span_bcal.yaml` 里的 3500 写在 `qat.sqat`(数据侧)和 `qat.qalora.gptq`(train.py 读)下,
而 `qat.sqat_permute.gptq.nsamples` 仍是 128——`make_matched_base.py` 读的是后者。所以要么派生一份配置把
`qat.sqat_permute.gptq.nsamples` 改成 3500、`batch_size` 改成 8,要么给脚本加 `--nsamples/--batch_size` 覆盖。
输出目录名用 `--tag` 区分(默认 `Llama-2-7B_int{b}_{g}_asym_sqatcal` 已被 BoolQ-only 的 matched base 占用,
脚本见到 `quantize_config.json` 就直接跳过,不会重建):

```bash
# 一次性,单 GPU;$LOTA_ENV 的 python(见 jobs/lota_matched_bases.pbs 里的 lota_py 包装)
lota_py baseline/LoTA-QAF/sqat/make_matched_base.py \
    --config configs/qalora_cs170k_int2_g32_ep1_span_bcal.yaml \   # 或派生后的 *_bcal_matched.yaml
    --out outputs/lota_bases --tag Llama-2-7B_bcal
# -> outputs/lota_bases/Llama-2-7B_bcal_int2_32_asym_sqatcal   (INT3 同理,用 int3_g64 的 bcal 配置)
```

然后训练/导出/评测走现成 pipeline,把基座显式传进去(`--base_dir` 同时意味着 `--skip_quant`):

```bash
bash runs/lota/run_lota_commonsense.sh \
    --config     baseline/LoTA-QAF/sqat/configs/lota_cs170k_int2_g32_ep1_span.yaml \
    --bits 2 --group_size 32 \
    --base_dir   outputs/lota_bases/Llama-2-7B_bcal_int2_32_asym_sqatcal \
    --with_base \                      # 顺便把裸基座评一遍:这是 LoTA-QAF 自己的 floor
    --note       "bcal matched base (balanced 3500, this repo's GPTQ grid); recipe unchanged"
```

先在 log 里确认两行:`[Data] calibration: 3500 records, sampling=balanced, task mix={...每个任务 ~437}` 和
`[GPTQ] Captured 438 calibration batches (3500 sequences, 470882 real tokens ...)`。没有这两行就是校准没换。

## 4. QWHA:两条路,优先第一条

QWHA 的 `baseline/QWHA/sqat/quantize_base.py` 走 optimum/gptqmodel 的 `GPTQConfig(dataset="wikitext2")`
(128×2048 通用文本)——这正是 E 臂那种会丢模板的校准,不能用于 bcal 行。

**路线 A(推荐,网格与 QA-LoRA/SALT-Q 完全一致)**:复用 LoTA 的打包方式——用 `make_matched_base.py`
(或照它写一个 QWHA 版)产出 GPTQModel 格式的基座,然后让 `init_adapter.py` / `train_commonsense.py` 从那个目录加载
(`gptq_base_dir(...)` 指到它,或加 `--base_dir`)。要核对 `quantize_config.json` 里 `sym=False, desc_act=False,
group_size, bits` 与 QWHA 加载路径的期望一致;AdaAlloc 初始化用的 X^T X 也应改成同一套 balanced 记录
(`load_calibration_data`),不要再用 wikitext2——初始化统计和量化统计来自同一分布才是单变量。

**路线 B(退路)**:保留 optimum 路径,但把 `dataset` 换成我们的 balanced 记录:用
`src.data.load_calibration_data(cfg, tokenizer)` 取 3500 条,把每条 `input_ids/attention_mask` 作为
tokenized 样本列表传给 `GPTQConfig(dataset=[...])`(optimum 接受 tokenized dict 列表)。这样校准分布一致,
但 GPTQ 实现(gptqmodel 的 act-order/damp 细节)与我们不同,行的解释里要注明"同校准、不同 GPTQ 实现"。

两条路都要在 log 里打印任务混合与真实 token 数,和 §3 一样核对。

## 5. 提交与记录

- 作业模板:`jobs/cs_lota_int2_span.pbs`(1 GPU,24 h)、`jobs/cs_qalora_int2_span_bcal.pbs`(4 GPU,`-q normal`,12 h)。
  固定套路:`cd "$PBS_O_WORKDIR"` → `conda activate saltq` → vllm 环境 import 自检 → `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`
  (计算节点无外网,数据集/权重要先在登录节点 build 进 HF 缓存)→ 调 `runs/<method>/run_*_commonsense.sh` →
  `echo "rc=$?"`。4 GPU 作业会路由到 g3,**g3 每用户同时只跑 1 个作业**;1 GPU 作业可走 g1/gdev。
- 每次派生配置**必须改 `training.output_dir`**(pipeline 的 `--output_root/--output_dir` 只影响查找,不改写入位置);
  基座目录另起名字,不要覆盖别人的基座。
- `conda run -n <env> python - <<EOF` 不转发 stdin,脚本写成文件再跑。
- 评测由 pipeline 自动跑 `runs/eval_vllm.sh` 并用 `scripts/collect_saltq_results.py --filter lota|qwha` 折进
  `results_saltq.csv`(`results/` 不进 git,CSV 才是记录);QWHA 的 run 脚本若没有 collect 步骤,手动跑一次
  `python scripts/collect_saltq_results.py --results_dir results --csv results_saltq.csv --config <yaml> --filter qwha --note "..."`。
- 结果填进 `SALTQ_experiment_tracker.md` 的「bcal」小节对应 LoTA-QAF / QWHA 行;Gap = (Avg − floor)/(77.75 − floor),
  LoTA-QAF 用自己 `--with_base` 评出的 floor,QWHA 若走路线 A 用 bcal floor D。只与 bcal 行比较,不要和旧格子混排。
- 提交 configs/jobs/CSV 到 git;不要动 `results/`、`outputs/` 之外别人的未跟踪文件。
