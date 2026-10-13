# 2026-10-04 / 2026-10-05 梯度异常诊断

**原 32 卡任务的非有限梯度根因尚未确定。** 四卡反转顺序实验已在第54次更新前触发同类非有限总范数错误，并保存完整现场；尚未确认与历史故障具有同一根因。
已确认存在极端 state 数据和旧 FlashQLA 的 gate 梯度精度偏差，但现有对照不能把任一项认定为原故障原因。
生产代码没有修改；本目录包含独立诊断脚本、日志、配置和产物。

原任务的直接证据：

- `workspace/gerty-qwen-streaming/20261003T182042Z_cotrain_5k_vlm_fsdp`：完成 87 次更新，
  第 88 次更新前在梯度裁剪处失败。step 40、80 日志中的区间平均范数已经约为 `8.99e16`、`1.50e17`。
  loss 保持有限。该错误只能证明总范数非有限，不能区分梯度元素非有限与 float32 范数计算溢出。
- 32 个 rank 的错误 JSON 只保存各自最后一个 microbatch 的 metadata，缺少此前 3 个累积 microbatch、
  当时的模型权重及 RNG，因此无法精确重放原第 88 步。
- acc1 任务的最早可见故障是 `qb-prod-gpu632` 的 FSDP `REDUCE_SCATTER` watchdog 超时，
  其他节点停在 HSDP `ALLREDUCE`。未找到更早的 Python OOM/CUDA 异常；超时根因也未证明。

本次实测环境为 `yhtian-h200-5`，4×H200，代码 HEAD `4cdcf3b`。
以下训练均使用 5k 配方、每卡 24576-token packing 预算、4 次梯度累积，保留原 100000 步 scheduler horizon。

| 测试 | 结果 | 证据 |
| --- | --- | --- |
| 四卡纯 FSDP，虚拟 epoch 1M、workers 2、关闭 EMA | 120 步成功；最大 float64 全局范数 790.20 | `train3.log`, `fsdp4_summary.json`, `training/` |
| 四卡 HSDP，2 replicas × 2 shards；同一数据轨迹 | 120 步成功；最大原生范数 402.81 | `train_hsdp.log`, `training_hsdp/` |
| 四卡纯 FSDP，恢复 1B epoch、workers 4、EMA 0.999 | 120 步成功；最大原生范数 472.66 | `train_original.log`, `training_original/` |
| 原错误记录的 32 个 pack、217 个样本，初始权重重放 | 全部梯度有限；范数 93.40–230.13 | `pack_rank*.json`, `replay_bf16.log` |
| 这 32 个 pack，用诊断 step-120 权重固定 RNG 比较两后端 | 64 次反向均有限；最大范数 34.44 | `trained_replay/`, `replay_trained.log` |
| 12 个极端 state 样本：单时刻/完整历史 × 原 state/裁至 ±10 × 两后端 | 96 次反向均有限；最大范数 847.35 | `outlier_candidates.json`, `outlier_replay/` |
| 强制触发现场捕获，然后加载保存的权重、真实 batch 和 RNG | 四个 rank 均保存全部 4 个 microbatch；32 次双后端重放成功 | `capture_smoke/`, `replay_capture_smoke.log` |

三次 120 步训练的 checkpoint 均有 `_SUCCESS`，退出码均为 0。
强制捕获测试的退出码 1 是主动抛出的诊断停止信号；其重放退出码为 0。
上述三次短训练和重放均已退出；2026-10-05 已另行启动四卡续训，见下文实时日志。

这些结果有明确边界：四卡无法覆盖原 4×8 卡跨节点通信；最初两轮改变了虚拟 epoch 和 worker 数，
第三轮恢复了这些设置，但 world size 仍不同。初始权重及 step-120 pack 重放重新进行了数据处理，
并非原任务的精确增强和随机数。第一个纯 FSDP 诊断还在原裁剪前额外同步并检查了梯度；
后两轮先运行生产裁剪，再诊断，避免额外的裁剪前同步。小算子测试曾短暂共用 HSDP 的 GPU 0，
因此这些运行不是严格的性能或 CUDA 时序对照。

数据扫描：

- 对配方引用的全部 **39 个机器人数据源、496114 个 episode、600428307 个时间点**，
  扫描了 `action*` / `observation*` 数值字段：未发现 NaN、Inf 或 masked 缺失值。
- 对同样全部时间点，经真实字段映射、unification、normalizer 扫描 state：未发现非有限值。
  任一 state 维度绝对值大于 10 / 100 / 1000 的时间点分别有 **408628 / 1278 / 895** 个。
  最大绝对值为 **58815.91015625**。
- DROID 在诊断脚本中最初遗漏了标量 gripper 的 `[T] -> [T,1]` 转换，导致该脚本失败；
  已改为调用训练数据路径的 `read_temporal_feature`，并补跑两个 DROID 源至完成。
  该错误属于诊断脚本，不属于训练数据故障。
- 原错误 metadata 涉及的 164 个机器人 episode 也单独扫描了完整数值轨迹，共 257051 个时间点，全部有限。
- 扫描不包含视频逐帧解码、VLM 文本语义审核，也不等同于对所有可能的 action 相对化/增强组合做训练。

完整统计在 `data_audit_summary.json`、`all_numeric/`、`all_states/`。
例如下列 state 会进入模型，动作裁剪不会处理它们：

| dataset | episode UUID | 原始时间点索引 | 时间 | normalized state |
| --- | --- | --- | --- | --- |
| `coin_agilex_cobot_decoupled_magic_4cam_30fps__7a937cd691be` | `7210db69-bbbd-5efd-b27c-f0cdf0f83483` | 430 | 14.3333 s | dim 0，绝对值 58815.91 |
| `coin_agilex_cobot_decoupled_magic_3cam_30fps__031240cb90fb` | `7d886ac0-ca3f-585f-8234-b55b90f3b848` | 2160 | 72 s | dim 0，绝对值 58815.91 |

这属于数据质量问题，不能因为本次没有溢出就视为正常数据；但也不能据此宣称它导致了原故障。
所有 12 个极值样本的裁剪与后端对照均未重现原来的巨大梯度。

FlashQLA 核查：

- 当前安装的是 `flash_qla==0.1.2`。其 Hopper 融合反向没有包含上游 gate 梯度高精度转置修复。
  上游说明见 https://github.com/QwenLM/FlashQLA/pull/28 ，修复提交为 `6e5d2ea153`。
- 固定输入与上游梯度的小算子对照中，强衰减条件下 `dg` 与 FLA 的相对 L2 差异最高约 18.6%，
  但没有出现非有限值或巨大梯度。结果见 `gdn_compare.jsonl`。
- 160 次长短序列混合的算子压力测试也全部有限，见 `gdn_stress.log`。
- 当前包还缺少上游 Hopper `fence.proxy.async` 修复 `c4d72349d2`。
  这是待验证线索，不是已证明的原故障原因；本次没有修改或升级共享环境中的依赖。

后续按用户指定继续使用四卡，不将 32 卡资源作为前置条件。2026-10-05 已启动全量相对 action 范围扫描，并已从原设置的四卡第 120 步 checkpoint 恢复模型、优化器、scheduler、loader、RNG、EMA，续训目标为第 1000 步，捕获首次大梯度。以下 32 卡入口仅作备用记录。
已准备 `train_config_32.yaml` 和 `scripts/run_train_probe_32.sh`：

- 保留原 1B epoch、workers 4、EMA 0.999、acc 4、每节点 shard size 8、100000 步 scheduler horizon。
- 最多运行 120 步。首次原生梯度范数大于 `1e8` 时，各 rank 保存模型及当前全部累积 batch/RNG，并停止。
- 原生裁剪抛错时，额外保存每个参数本地梯度的 float64 范数、绝对值最大值和非有限元素数量，
  用于区分范数溢出与梯度本身非有限。
- 普通训练步不在原裁剪前增加诊断 collective。捕获之后的 barrier 只服务于异常停机。
- 捕获和重放已在四卡强制触发验证；32 卡配置已通过 `TrainConfig` 校验，但尚未做 32 卡运行验证。

在原 32 卡任务相同的 PET_* 分布式环境中，替换启动入口为：

```bash
cd /inspire/qb-ilm/project/gjjproject/public/yhtian/projects/gerty
SECRET=private RUNMODE=python bash tmp/ai_tasks/20261004_gradient_diagnosis/scripts/run_train_probe_32.sh
```

当前平台查询能看到原失败任务的 `训练区-H200-1号机房-cuda13.2` / 每节点 8 GPU 资源，
但实时 job quota 列表没有对应 H200 8 卡配额，因此没有提交新任务。
资源查询记录在 `original_job_status.json` 和 `job_quotas.json`。


## 2026-10-05 全量范围扫描完成

`audit_all_actions.exit` 与 `audit_all_states_signed.exit` 均为 0；核验三份全量扫描的数据源集合、
episode 数、时间点数完全一致：39 源 / 496114 episode / 600428307 ticks。
`signed` 脚本把有符号极值补充写入原 `all_states/`，保留了原计数和异常样本字段。

| 检查对象 | 全量最小值 | 全量最大值 | 非有限值 |
| --- | ---: | ---: | ---: |
| 归一化 state | -520.0151977539062 | 58815.91015625 | 0 |
| 相对化、归一化后 action，裁剪前，有效维度 | -10066331 | 10066329 | 0 |
| 模型接收的 action，裁剪后 | -1.5 | 1.5 | 0 |

原始字段的单位和维度不同，不把全部字段压成一个无物理意义的共同范围。
`raw_ranges.csv` 提供各 dataset/field 的真实最小值、最大值及对应 episode UUID；
`normalized_ranges.csv` 提供各源 state/action 的归一化范围。
完整总计见 `complete_range_summary.json`。

action 扫描以每个源时间点为当前 tick，检查长度为 `floor(action_horizon_sec * frequency + 0.5)`
的有效未来窗口。使用真实字段映射、unification、relative_mask 和 normalizer；
通过滚动最小/最大值计算线性归一化后的精确范围，排除 padded action 维度/时间点。
奇偶窗口、窗口超过 episode 长度，以及每个源首个 episode 的三个显式窗口均通过直接计算核对。
有至少一个未来 action 超过 ±1.5 的当前 tick 共 86583453 个；
超过绝对值 10 / 100 / 1000 的当前 tick 分别为 2046566 / 1296875 / 1208864。
这些是重叠窗口的当前 tick 数，不是独立异常 action 元素数。

两个具体异常来源：

- `gow_r1lite`，episode `881bd637-b6fd-5062-8692-eec0ef7b10af`，
  当前 tick 1139、未来 tick 1153、统一后的 action dim 16（15 Hz）。
  该维度不做 relative subtraction，映射后 action=-0.6000000238418579，
  统计量 q01=q99=0。当前 `GroupedNormalizer` 使用 float32 epsilon 替代零宽度，
  得到 -10066331，裁剪后 -1.5。这证明归一化放大来源，但不能证明它导致原梯度故障。
- `coin_agilex_cobot_decoupled_magic_4cam_30fps__7a937cd691be`，
  episode `8343b335-c176-50f9-9617-88b1a3ccb325`，当前 tick 134、未来 tick 149、dim 0。
  state=33071.62109375，action=-0.8146206736564636；相对化后归一化 action=-143200.0625。
  原始 state 本身已经异常大。更多见 `all_actions/*.json` 的 `worst_episodes`。

## 2026-10-05 内核检测与续训

- `gdn_mixed_decay.jsonl`：96 组强/弱衰减混合、相似 key、beta=1、长短变长序列，
  每组固定输入对比 FlashQLA 与 FLA。全部有限；FlashQLA 的输出/输入梯度最大绝对值 47.34，
  gate 梯度相对 L2 最大差异约 1.74%。未复现巨大梯度。
- `gdn_sanitizer.log`：短序列 CUDA racecheck 报告 28 个分析错误，位于 `kkt_solve` 的
  16×16 对角求逆循环。`gdn_fence_sanitizer.log` 中仅加上游 kkt fence 后仍为 28 个。
- `qla_sync_trial/` 为独立包副本：在循环写回后加 `T.sync_threads()`，
  `gdn_sync_sanitizer2.log` 降至 0 hazards，退出码 0；两个测试的输出及梯度最大值保持一致。
  首次同步版 sanitizer 启动时工具进程 SIGSEGV，原日志保留，第二次完整完成。
  生成的代码还含有不写回结果的重复读取线程；因此不能凭 racecheck 报告认定它造成原 Inf。
  共享安装包没有改动。
- `gdn_long_sanitizer.log`：排除上述已知 kkt 内核后，对 [1025,63] / [4097,130]
  长序列测试其他内核，racecheck 为 0 hazards。
- `gdn_initcheck.log`：报告融合反向读取 `h` 时存在未初始化全局显存，
  300 秒限时内只完整跑完第一组，退出 124。生成代码见 `bwd_*.cu`、`h_*.cu`；
  `h` 由 TMA store 写入。
- `gdn_poison_direct.log`：直接调用 FlashQLA，启用 PyTorch 对新分配浮点显存填 NaN；
  确认 `torch.empty` 为 NaN，并检查反向使用的整个 `h`。
  两组 `h` 均没有非有限值，输出/梯度均有限，退出 0。
  因此 initcheck 报告尚未构成数据污染的证据，可能涉及其对 TMA store 的追踪；不能据此宣布根因。
- `train_extended2.log`：四卡保持 epoch 1B、workers 4、EMA 0.999，已从第 120 步完整恢复，
  目标第 1000 步，实时裁剪范数见 `training_extended/rank*.jsonl`。
  第一启动日志 `train_extended.log` 是在训练前为修正 checkpoint 绝对路径而主动终止的。
  原生梯度范数首次超过 1e8 或裁剪失败时，会保存模型、全部 4 个累积 batch 和 RNG。
  临时 probe 后续加上的逐参数 spike 统计只对下次新进程生效，当前续训仍使用启动时版本。

四卡长训期间部分短内核测试共用 GPU，以上不是性能对照。
原 32 卡巨大梯度和超时的根因仍未证实；继续四卡实验，不以 32 卡资源为前置条件。

FA3 packed action self-attention 追加对照：`fa3_probe.log` / `fa3_probe.jsonl`，24 组短/长 action slot、GQA 8:2、head_dim 256、BF16，对照逐 slot FP32 math SDPA；全部前向/反向有限，最大相对 L2 误差 0.002648，退出码 0。FlexAttention 探针已准备，尚待终端恢复后启动。

FlexAttention 追加对照已完成：`flex_probe.log` / `flex_probe.jsonl`，12 组真实 expert joint mask（多 slot 同 segment、部分 prefix、padding、确定性开关交替），实际 compiled `_joint_attention` 对照逐 slot FP32 math SDPA；全部前向/反向有限，最大相对 L2 误差 0.003106，退出码 0。

终端偶发立即退出 124 已追到 Jupyter terminal API HTTP401，而非训练停止。`local_terminal_retry.py` 是工作站上的临时 CLI 包装器：从 IDE UI 刷新正常认证路径后再创建终端；不改已安装 CLI，不保存或输出 token。

## 2026-10-05 原始数据流重建与下一组实验

- `reconstructed_data/rank0/comparison.json`：原 rank 0 的第 352 个 microbatch 与故障快照的
  6 条 metadata 完全一致，包含 episode UUID、数据集和样本索引、时间窗口。
  同次运行保存 `microbatch348.pt` 至 `microbatch351.pt`，补齐第 88 步的四个累积输入。
  原训练没有保存处理后张量，因此这里证明的是 metadata 对齐，不能称像素逐位验证。
  `reconstruct_rank0_2.exit=0`；结束后的 worker 析构报错不是原始梯度故障。
- rank 8、16、24 的相同重建正在运行，覆盖原四个节点各一个 rank；日志各自独立保存。
- 四卡续训最新核对完成 619 步，从第 120 步恢复以来最大范数为 24.589191436767578。
  原任务前 30 步记录的窗口均值约 100，巨大范数首次见于第 40 步统计窗口；尚不知道具体首发 microbatch。
- `train_virtual32.py` 将 32 个原始 loader 流分配给四张卡，每卡八个流；acc32 保持每次更新 128 个 pack。
  各逻辑 rank 独立保留 loader RNG 与模型 CPU/CUDA RNG；在 microbatch351 再核对全部 rank 的故障 metadata。
  已通过八个真实 StatefulDataLoader 的轮转及中途恢复测试，以及四卡/acc32 配置校验。
  尚未完成 GPU 训练验证；四卡 FSDP 不等于原 HSDP 通信拓扑，不能承诺浮点轨迹逐位相同。
- `scripts/queue_virtual32.sh` 已在目标容器后台运行：仅在当前长训退出 0、step1000 checkpoint 有 `_SUCCESS`、
  三个 rank 重建均退出 0 后启动上述实验。任一前置失败会停止衔接。
  新训练首次范数超过 1e8 时保存模型、全部累积输入、RNG 和裁剪后逐参数统计；不修改正式训练代码。

回放精度补充：旧 `replay_spike.py` 把整模型转为 BF16，再独立回放每个 microbatch，
它不是原训练 FSDP/累积路径的严格数值复现。新增 `replay_fsdp_capture.py` 保留原参数存储精度、
FSDP mixed precision、reshard 设置、全部累积输入及各 microbatch RNG；对已保存正常现场重复两次，
要求梯度范数与捕获值 125.77989196777344 的相对误差小于 0.001。
该 GPU 校验安排在 virtual32 启动脚本最前面，尚未执行，不能报告通过。
virtual32 同时新增对已重建各步 metadata 的核对，以及故障现场中的逻辑 rank/microbatch 编号。
修改后的 CPU 轮转、恢复及配置检查已在目标容器再次通过。

原 rank 8、16、24 重建现已全部完成，退出 0；加上 rank 0，第 352 个 microbatch 的
33 条 metadata 均与各自故障快照完全一致，四个 rank 的第 348..351（零基）批次共 16 个输入全部保存。
`inspect_reconstructed_batches.py` 逐个加载它们，检查 109 条样本的处理后张量：

- state：[-37.92036056518555, 19.212108612060547]，无 NaN/Inf。
- action：[-1.5, 1.5]，无 NaN/Inf。
- pixel_values：[-1.1923279762268066, 1.1897752285003662]，无 NaN/Inf。
- 最大 state 偏离位于原 rank0 microbatch348 的第24维，来自 `gow_r1lite`，
  episode `f61a0ab6-ae33-5881-95c4-971cb8b967e7`，时间窗口 [58.65108447549085, 88.11513069380187] 秒。
  这是故障步第一个累积 microbatch 中的输入，尚无证据证明它导致梯度异常。

检查退出码 0，完整范围和来源见 `reconstructed_data/step88_input_ranges.json`。
该检查覆盖原 32 个 rank 中四个 rank 的完整累积输入，不代表已检查全部 128 个原始 batch。

实验优先级调整：原数据流重建校验完成后，主动在第 728 个更新结束长训，优先运行原128-pack更新的复现。
第120步之后的608个更新最大范数24.589191436767578，未触发1e8捕获阈值。
先停止等待队列PID521058，再SIGTERM训练torchrun81694，已核验四个rank退出及训练GPU显存释放。
`train_extended2.exit=1` 是本次主动结束的结果，不是梯度故障；没有生成第1000步checkpoint。
直接启动 `run_train_virtual32.sh`（启动壳PID2828137），先执行FSDP正常现场回放校验，成功后才开始virtual32训练。

FSDP回放现已通过：所有四个rank的两次范数分别为125.78406524658203、125.76758575439453，
相对捕获值125.77989196777344的误差分别3.3179e-5、9.7839e-5。
这支持恢复精度/累积路径的回放有效性，不是逐位确定性证明。
随后virtual32正式进程已启动，workspace为`gradient-debug/20261005T140626Z_virtual32-four-card`，
torchrun PID2884697，四个rank PID2884700..2884703；尚未核验第一个更新。

virtual32首个更新现已核验：14:19:58 UTC完成，loss16.473324552178383，范数121.07127380371094，
每次更新128个pack。首步603.5秒中510.9秒为数据加载，主要是依次启动32个原始四worker loader；
原rank0/8/16/24已处理批次的metadata检查未失败。四卡范数日志在`virtual32/rank*.jsonl`。
继续观察原任务首次巨大范数的31..40步窗口，尚未复现巨大梯度。

清理了第一次失败的`gdn_sync_sanitizer.log`遗留进程：其TreeLauncherSubreaper/Python PID314615/314621
在sanitizer前端SIGSEGV后仍存活。已确认输出路径后终止，核验GPU仅剩新的四个训练rank。

virtual32前10步完成后，按原CLI的10步窗口归约规则重组日志：计数取每步平均、min/max取窗口极值、
packing均值按计数加权，其余指标取均值。`compare_virtual_trajectory.py` 输出
`virtual32_trajectory_comparison.json`：

- 所有数据统计（AR tokens、FM slots、各类segments、packing计数/长度）与原窗口一致，误差小于1e-8。
- 平均loss：16.508696857839823，对照原值16.508698920160533，绝对差2.062320710649601e-6。
- 平均范数：121.71500930786132，对照原值121.32548141479492，差0.38952789306640057，约0.32%。

这支持当前数据流及早期loss轨迹与原任务接近，不等于不同通信拓扑下的梯度逐位一致。
继续运行至原31..40步巨大范数窗口及其后；目前尚未复现。

11..20步窗口也已完成：数据统计仍一致；平均loss16.10728973597288，对照16.106995331496，
差0.0002944044768788956；平均范数96.32916412353515，对照96.65824966430664，差约0.34%。
截至第20步没有捕获巨大梯度，模型早期轨迹尚未显著偏离原任务。

21..30步窗口继续匹配全部数据统计；平均loss15.01939088255167，对照15.020363873243332，
差0.0009729906916611242；平均范数107.06899986267089，对照107.69243240356445，差约0.58%。
截至30步尚无巨大梯度，即将进入原31..40步异常窗口。

31..40步窗口完成：全部数据统计仍一致；平均loss13.911424518376588，对照13.856496945768594，
差0.05492757260799408；平均范数189.93295068740844，原窗口为8.990357485382538e16。
没有复现原来的巨大范数，但第39/40步发生较小跳变：872.5304565429688/868.7254638671875，
对比第38步17.317018508911133和第41步21.948457717895508。
运行中的1e8捕获阈值没有保存这些较小跳变的权重；不能将它们直接认定为原故障。
临时train_probe现支持环境变量DIAG_CAPTURE_NORM（默认仍1e8），仅对未来进程生效，
供后续以500左右阈值捕获早期放大现场并做模块级回放定位。

下一轮已排队（PID1993626）：当前virtual32正常结束且第120步checkpoint有`_SUCCESS`后，
启动四卡HSDP2×2、同32个原始数据流，捕获阈值500，输出到`virtual32_hsdp/`。
当前训练继续覆盖原第88步故障位置；下一轮尚未启动。
`prepare_virtual32.py`新增DIAG_SHARD_SIZE环境参数，默认4，下一轮设2；正式训练代码没有改动。

41..50步窗口仍匹配数据统计：平均loss13.636255140230059（原13.656454186141492），
平均范数28.88912162780762（原87.68929738998413），未出现非有限范数。

新增隔离对照包`qla_full_fence_trial/`，包含上游c4d72349d2的全部Hopper fence改动：
cp_bwd 8、cp_fwd 1、fused_bwd 17、fused_fwd 6、kkt_solve 1、prepare_h 9，共42处。
已核对实际diff只有fence插入、检查fused_bwd偏移上下文，并通过AST解析；源文件与副本哈希在manifest.json。
尚未做该完整补丁的GPU测试，没有修改已安装包，也没有将它认定为修复。
另补充原NaN分配测试的限制：`gdn_poison_direct.py`在backward内检查h时调用CPU同步，
因此有限结果不能排除时间顺序/可见性问题。

51..60步数据统计继续一致；平均loss13.650227036327124（原13.584115746617316，差0.0661113），
平均范数113.22356405258179（原37.246495485305786）。第58步范数480.86871337890625，
第59步162.60189819335938，第61步回落14.054088592529297，均有限。
这说明当前梯度轨迹已与原任务不同；数据一致和loss接近不能替代原巨大梯度的复现证据。

71..80步同样没有复现巨大范数：数据统计一致，平均loss13.328308344632386（原13.354251974821091），
平均范数7.344029521942138（原1.5042076356609312e17）。截至80步全部八个10步窗口的数据统计均匹配。
继续至88步核对全部32个原rank的最后故障microbatch metadata，再完成有界120步实验或在捕获时停止。

第88步已完成：loss13.13851910829544、范数5.521839141845703；32个逻辑rank的217条故障metadata全部相等。
证据在`virtual32/data_match_virtual_rank*.json`和`original_step88_metadata_summary.json`。
它证明来源/窗口/顺序对齐，原始处理后张量及模型权重未保存，因此不能证明逐位复现原计算。

完成该故障位置核对后，为优先拿到梯度跳变现场，主动在第90步结束纯FSDP对照并提前启动HSDP2×2。
90步最大范数872.5304565429688，未复现巨大梯度。先停止队列1993626，再停止torchrun2884697，
已核验四个rank退出和GPU释放；`train_virtual32.exit=1`来自主动SIGTERM，没有第120步checkpoint。
新启动壳PID4019189，日志`train_virtual32_hsdp.log`，捕获阈值500；初始FSDP4 smoke回放阶段后才是HSDP2×2训练。
上一轮smoke结果已从原日志归档为`capture_smoke/fsdp_replay_virtual32_rank*.json`，避免通用输出文件被新回放覆盖后混淆。

## 2026-10-06：HSDP 有限尖峰捕获与对照

原始 Inf 的根因仍未证明，生产代码未修改。以下结果解释的是一个约 827 的有限尖峰，
不能将它等同于原始训练的约 1e17 梯度窗口或第 88 步非有限范数。

HSDP 2×2、32 条逻辑数据流在第 39 次裁剪触发阈值 500；此前完成 38 次 optimizer update。
现场范数 827.4854125976562，四个 rank 的完整权重、32 个累积 batch/RNG 和逐参数梯度统计已保存到
`virtual32_hsdp/`。诊断 barrier 后主动抛错退出，`train_virtual32_hsdp.exit=1` 是预期的捕获停止。
所有梯度元素有限，两个 HSDP 副本的统计一致。backbone 第 0 层占梯度平方和约 30.855%，
action expert 合计约 0.001154%。源码确认 FM 使用 detach 后的 backbone K/V。

相同现场的完整 HSDP 重放均成功退出，四个 rank 均有限：

| 后端 | 第一次范数 | 第二次范数 |
| --- | ---: | ---: |
| 当前 FlashQLA | 827.42749 | 827.40143 |
| 42 处 Hopper fence 隔离补丁 | 827.46906 | 827.43854 |
| FLA | 841.96558 | 841.89325 |

原版与 fence 版本的全部逐 microbatch loss 完全一致；FLA 的最大单批 loss 差异为 0.06260。
版本路径、启用状态和六个 kernel 文件哈希均记录在重放日志。
证据：`virtual32_hsdp/backend_comparison.json`、`replay_hsdp_{installed,full_fence,fla}.log/.exit`。
隔离补丁未消除有限尖峰，未据此升级共享依赖或声称修复原故障。

逐 microbatch 诊断保留 FP32 存储、BF16 计算、HSDP 归约及原始 1/32 loss 缩放；
额外范数 kernel 改变执行时序，因此它用于定位，不替代上面的完整重放。
总范数仍为 827.45935。第 0 层 out_proj 的 32 次增量范数为 4.9234–16.4776，
总和 297.6942，最终累积范数 276.9559；最终范数/增量范数和为 0.93034。
最大增量只占该和的 5.535%，平均增量与之前累积梯度的夹角余弦为 0.93163。
这一个参数的尖峰不是单个 microstep 主导；不能据此把某条 UUID 判为坏样本。
`microbatch_gradient_ranking.json` 保留了全部 128 个 pack 的来源信息。

针对 microstep 10 的四个 pack，归一化直接反向 hook 检查发现多层 Gated RMSNorm 局部放大数百至千倍。
但保存的零更新权重也存在类似尺度；例如第 22 层初始约 1092–1154 倍，训练后约 1251–1361 倍。
因此高局部放大不是已证明的新增故障。初始权重来自 `capture_smoke`，核验 completed_steps=0，
逐参数从原四分片收集再分为当前 HSDP 2×2；保持当前输入与 RNG。

同一 microstep、同样 1/32 loss 缩放的权重替换结果：

| 权重 | 梯度范数 |
| --- | ---: |
| 捕获的训练后权重 | 80.74155 |
| 全部恢复初始权重 | 3.99398 |
| 仅恢复初始词嵌入/输出头 | 67.88967 |
| 仅恢复初始 decoder 层及最终 norm | 4.00241 |

运行时确认词嵌入和输出头共享权重；替换记录包含全部实际 state_dict key，避免遗漏别名。
结果支持这个有限尖峰主要随 decoder 状态变化，尚不能解释原始 Inf。
证据：`normalization_initial_comparison.json`、`weight_ablation_summary.json` 及各 rank 原始报告。

已启动 `virtual32_hsdp_large`：相同 HSDP 2×2 和原始逻辑数据轨迹，捕获阈值改为大于 1e8，
计划运行至 120 步并覆盖原始失败步 88；每 30 步保存完整 checkpoint。
启动后仍先执行 FSDP4 smoke 重放；实际 HSDP 训练、checkpoint 成功及数据对齐需从新日志逐项验证。
日志为 `train_virtual32_hsdp_large.log/.exit`，启动 PID 记录在 `launch_hsdp_large.pid`。

## 裁剪报错语义的 CPU 对照

远端 Torch 2.11.0+cu130、VeOmni FSDP2 默认裁剪函数，使用 CPU 的 1024 个 FP32 梯度元素：
每个元素 1e17 时裁剪成功；每个元素 1e18 时，全部元素仍有限、FP64 范数约 3.2e19，
但原生裁剪抛出相同的 `total norm ... is non-finite` 错误。
这验证了报错本身不能证明梯度元素非有限；它不证明原始 GPU 故障就是求范数溢出。
证据：`probe_norm_overflow_cpu.py`、`probe_norm_overflow_cpu.json`。未据此修改生产裁剪。

## HSDP 原失败位置对照

`virtual32_hsdp_large` 已通过第 88 步，梯度范数 5.454963684082031。
32 条逻辑数据流最后一组 microbatch 的 217 个样本 metadata 与原错误现场逐条一致。
前八个十步窗口的数据统计也一致，但原来的两个约 1e17 范数窗口均未复现。
这验证样本来源与顺序对齐，不代表重建了原始张量、模型状态或 4×8 HSDP 的浮点归约顺序。
原 Inf 根因仍未知。证据：`virtual32_hsdp_large/step88_metadata_summary.json` 及逐 rank 对照。

HSDP 对照最终完成 120 步、退出码 0，四个 rank 全程范数一致且有限。最大范数 892.8328（第 39 步），
最后一步 12.01448。第 30/60/90/120 步 checkpoint 均通过 CPU 状态结构与 loader 边界检查。
随后启动第 30→32 步实际恢复验证；启动不等于恢复验证已通过。

## 实际恢复验证发现的预处理 RNG 缺口

第 30→32 步实际恢复在第一个 batch 的 metadata 断言失败，尚未完成任何更新。CPU 单 loader 复现：
microbatch 120/123 的样本组合不同，121/122 完全一致。保存 buffer 的四个 worker 均不足 packing 阈值，
恢复后需要继续预处理新样本。`ProcessedSampleDataset` 不保存预处理 RNG，processor 的随机视角/分辨率/丢弃会改变长度。
使用真实 VeOmni native worker packing 的小型随机长度实验，保存 12 个 batch 后续跑 8 个：
默认恢复不一致；隔离实验在 packing state 中保存并恢复 Python/NumPy/Torch RNG 后，8 个全部一致。
这确认一个恢复机制缺口，完整真实数据修复尚待验证，也不是原始 Inf 的因果证明。

真实数据后续验证通过：重新连续读取原逻辑 rank0 的120个 pack，逐条 metadata 与原序列一致，
保存含 worker RNG 的新状态后，由全新进程恢复；后续8个 pack 的 metadata 及每个 pack 的16个张量字段
均逐字节一致。create/resume 主进程均退出0，worker 析构仍打印 C++ abort，单独保留该清理问题。
证据 `real_rng_resume/comparison.json`、`create.json`、`resume.json`。候选依赖补丁 `veomni_worker_rng.patch`
尚未应用到共享依赖；旧 checkpoint 没有保存的 RNG，不能据此声称可精确恢复。该验证使用完整 buffer 保存，
不覆盖 save_by_idx 的随机重新处理。原始 Inf 的根因依然未知。

同一有限尖峰现场的12次完整 HSDP 重放全部有限，退出0。范数范围827.3171–827.4763，
相对波动0.01924%；每个 rank 的全部32个 microbatch loss 在12次间逐值完全一致。
这观察到小幅反向/归约数值变化，但没有复现巨大梯度或原始 Inf，不能归因为已证明的故障。

## 固定 microbatch 的注意力反向重复性对照

固定有限尖峰的权重、microstep10 的4个 pack、RNG、1/32 loss 缩放，分别重复24次反向，全部有限：

| Backbone full attention | 线性注意力 | 梯度范数范围 |
| --- | --- | --- |
| FA3 默认 | FlashQLA | 80.00147–81.06151 |
| FA3 默认 | FLA | 101.68198–103.04845 |
| FA2 deterministic=True | FlashQLA | 67.09041595458984，24次完全相同 |
| FA2 deterministic=True | FLA | 64.18704223632812，24次完全相同 |
| FA2 deterministic=False | FlashQLA | 66.44214–67.41230 |
| FA2 deterministic=False | FLA | 63.68174–64.82562 |

FA2 同线性后端的确定性开关对照中，所有前向 loss 完全相同；运行时验证实际传入的 deterministic 值。
这定位了该对照中的反向重复性差异，不能将正常非确定性误差视为原始约1e17梯度或Inf的原因。
切换FA3/FA2同时改变前向数值，不能把两者梯度差异全部归于反向。Action expert 中直接调用的FA3未被此配置替换。
原FA3 deterministic=True实际运行报 `Deterministic backward not supported for hdim 256.`，没有得到可用数值对照。
证据：`virtual32_hsdp/micro10_{repeat24,fa2det24,fa2plain24}_summary.json` 及逐rank结果/退出码。

继续提取固定microstep10全部18层线性注意力的真实q/k/v/g/beta、cu_seqlens及上游梯度，供原版和隔离fence版算子压力对照。
采集hook的CPU同步会改变时序；采集结果用于数值范围覆盖，不能排除竞态。生产代码与共享依赖保持不变。

真实输入算子压力测试完成：4rank×18层×24次=1728次反向/版本；当前FlashQLA与隔离fence版均退出0，全部输出与五类梯度有限。两版记录的absmax、非有限计数、相对FLA误差逐项相同（未记录所有张量的逐字节比较）。最大相对L2误差约1.98%，位于rank0的call9（从0开始编号）的dg。未复现原始Inf；证据`real_qla_stress_summary.json`与`real_qla_stress_{installed,full_fence}/rank*.jsonl`。

## 反转逻辑 rank 累加顺序的训练对照（进行中）

`train_virtual32_reverse.py` 在每个物理rank中反转8条逻辑数据流的处理次序，保留各自loader种子和RNG。
每次更新仍消费同样128个pack；HSDP2×2、原优化器/学习率与原生裁剪保持不变。
该实验寻找对浮点顺序敏感的失稳现场；FA3本身仍非确定性，因此仅有曲线差异不能归因于累加顺序。

启动重放校验通过，随后首个十步窗口完成：数据统计与原32卡任务一致（误差<1e-8），
平均loss16.50844816，对照16.50869892；平均范数121.262435，对照121.325481。
首步loss与之前正常顺序HSDP完全一致，范数121.076454，对照121.070686。
这只验证初始窗口，没有复现原始Inf。

当前日志`train_virtual32_hsdp_reverse.log`；进程和裁剪进度见`virtual32_hsdp_reverse/heartbeat.json`，
必须同时检查其中时间戳，旧心跳不是存活证明。远端监控每30秒从/proc核对训练进程命令行。
本地一次读取心跳JSON不完整导致观察器退出，重新读取确认远端训练仍存活；没有重启训练。

反转顺序实验已通过第31–40步窗口：数据统计仍与原任务一致；39/40步范数802.4913/801.3326，41步回落15.4177。窗口平均范数176.194，对照原任务8.99e16。没有复现原始巨大梯度；训练继续覆盖后续窗口。第30步checkpoint结构检查通过，但不能据此声称精确恢复。

## 四卡真实异常现场：第54次更新前

反转顺序训练完成53步后，原生裁剪报non-finite total norm退出1。所有4rank保存了权重及32个累积microbatch/RNG；重放程序已成功加载全部现场，completed_steps=53、逻辑microbatch212..215、每rank8条逻辑流各4批均通过核验。
逐参数统计：所有梯度元素有限，独立HSDP分片的FP64全局范数9.245048884e23，原生FP32求范数溢出。非backbone梯度范数2.846576。异常从第21层线性注意力的qkv/a/b/conv参数向前传播；该层out_proj、in_proj_z、MLP以及22/23层梯度均正常量级。不能用FP64裁剪绕过错误就宣称根因已修复。
当前原版后端的固定现场双次重放进行中。证据`virtual32_hsdp_reverse/failure_gradient_summary.json`、`failure_layer_gradient_norms.json`、`failure_capture_integrity.json`及`clip_failure_rank*.pt`。

### 后续重放与算子检查

上述双次重放已完成，范数分别约17.30689、17.30496，没有重现巨大梯度。保存的现场覆盖859条样本metadata。
重新计算并保存了第21层的128组输入；这是从失败权重重算的正常中间值，不是失败当时的中间缓冲区。
所有输入的token数均等于`cu_seqlens[-1]`，没有观察到上游dG尾部padding缺陷所需的触发条件。

原版FlashQLA的普通、空内存NaN填充、后台NCCL三组压力测试各完成32768次反向，均未出现异常；
最大输出/梯度绝对值0.2373046875，相对FLA最大L2误差约1.55%。这些阴性结果仍不能排除依赖执行时序的故障。
CUDA memcheck改为每个输入独立进程后继续检查；之前单进程检查的OOM属于诊断自身失败，不是原训练根因。
下一步在完整模型重放中捕获异常QLA backward的实际输入和输出。逐算子的检查会同步CPU，可能改变故障时序。

上述memcheck已完成：128组均退出0且报告0个错误。完整模型中间值检查也完成了16次重放，
每rank覆盖9216次QLA backward；梯度范数17.30375–17.31041，没有复现巨大梯度。
证据：`failure54_layer21_memcheck_summary.json`、`failure54_intermediates16_summary.json`。
这次hook记录的最大值包含未使用的`dh0`：Hopper kernel仅在`dht`非空时写入它，
当前路径没有使用这一输出，因此不能将该最大值解释为有效梯度最大值。后续训练hook已排除此项。

已启动完整训练过程捕获，workspace为`gradient-debug/20261005T233744Z_reverse-intermediates`。
保持反转逻辑rank顺序、HSDP2×2、原数据流及优化设置；在异常时保存算子的实际输入/输出和对应batch记录。
每个更新检查18×32次反向调用是否被截获。日志`train_reverse_intermediates.log`；
有效步数及存活时间戳见`reverse_training_intermediates_stats/heartbeat.json`。
逐算子同步可能改变故障出现概率，因此通过训练也不能单独证明根因已修复。

首个入口因DataLoader子进程重复执行模块初始化而失败；已添加`__main__`保护，
并通过实际子进程导入和完整训练步验证。当前有效workspace为
`gradient-debug/20261005T234353Z_reverse-intermediates`。

这轮训练已经通过第88步：第39/40步出现有限尖峰874.25/917.09，第41步回落25.66；
此前四卡失败的第54步范数18.25，原32卡任务失败的第88步范数5.256。
前80步各十步窗口的数据与packing统计均匹配原任务，原失败现场保存的32个rank的
最后microbatch metadata也全部匹配。这不证明所有原始处理后张量或执行时序一致。
截至第88步没有异常缓冲区捕获；该轮仍按预设运行到120步。

已准备但未启动`train_reverse_deferred_capture.py`：只保留第21层算子缓冲区，
在完整backward返回后检查，减少逐算子CPU同步。它仍改变张量生命周期及同步位置。
脚本和实际子进程导入检查通过，GPU运行尚未验证。当前运行结束并释放GPU后，
若仍无异常现场，再进行这个对照；不将阴性结果视为修复证明。

上述逐算子同步训练已完成120步并退出0：所有rank范数一致且有限，最大917.09，
最后一步11.72；总计276480次反向捕获检查，没有异常。GPU已释放。
完整结果见`reverse_training_intermediates_summary.json`。

后续优先采用`train_reverse_device_capture.py`，替代尚未启动的逐backward CPU检查方案：
GPU标记首次异常并条件复制实际缓冲区，到原生梯度裁剪后才读取CPU标志。
它仍增加GPU工作和持久分配，不能声称时序完全不变。
人工构造的大有限值/Inf/NaN测试通过，验证了首次异常在后续扩容后仍保留、跨stream读取有等待、
未写入的dh0不会误触发。一个真实重建QLA输入的前向和五类梯度与未观察基线逐元素一致；
该输入缓冲区占1348468864字节。测试见`device_capture_checks/`，属于捕获器验证，不是故障复现。
新四卡训练已提交，运行状态需查`train_reverse_device_capture.log`及对应PID/exit文件。

### 重放输入的 gate 衰减范围

CPU 扫描全部 128 组重建的第 21 层输入：gate 范围 `[-2.0020442, -4.96232118e-11]`，
无正值；beta 范围 `[1.44355e-7, 1]`。按每个序列内的 64-token chunk 累加，
最小 log 衰减为 `-59.46572`，对应 `1.49405e-26`，未越过 FP32 正常或非正规数下限。
这说明这些重建输入没有 chunk 衰减直接下溢的条件；它们来自成功重放，不能据此排除
失败当时的异常中间值或其他数值消减问题。见 `failure54_gate_ranges_summary.json`。

### 设备捕获实验的 checkpoint 中断

首轮设备捕获训练完成 30 步，梯度均有限，但在 checkpoint30 的 DCP `gather_object`
阶段，rank0 报 `NCCL Error 1: unhandled cuda error`。当时 GPU0 只剩 5 MiB 空闲；
显存不足是强线索，旧日志缺少底层 NCCL allocation 错误，尚不能作完整因果确认。
这是诊断过程的新中断，不是历史梯度故障。旧日志和捕获目录以 `.failed_checkpoint` 保留，
checkpoint30 没有 `_SUCCESS`。已清理该轮进程并确认 GPU 释放。
诊断入口现于 DCP 保存前记录 allocated/reserved/free，释放未使用的 PyTorch 缓存，
并开启 NCCL INFO；重新从初始状态运行，workspace 为 `20261006T025420Z_reverse-device`。
截至启动核验，新四个 rank 均存活，尚未验证第30步保存；生产代码未修改。

重启实验已通过第30步保存（03:37:33 UTC）：核验 `_SUCCESS=30`、模型和EMA的metadata、
各4个DCP分片及4份rank extra_state。rank0保存前 allocated=22.28 GB、reserved=146.83 GB，
空闲203.88 MB；`empty_cache()` 后空闲119.22 GB，保存成功。数据支持闲置缓存挤占NCCL显存
这一解释，但旧错误日志没有底层allocation详情。该调整仅位于诊断入口；原梯度根因仍未确定。
前30步的数据/packing统计仍匹配原任务，全部梯度有限，设备捕获尚未触发。

### 设备端捕获训练完成

`20261006T025420Z_reverse-device` 完成120步并退出0，最终checkpoint完整。
所有rank范数一致且有限，最大769.75（第39步），最后11.74；核验276480次总反向调用，
其中实际检测第21层输出15360次，没有异常捕获。第88步32个逻辑rank的最后microbatch
metadata与历史记录全部一致。这次仍未解释原故障，也不构成修复证明。
汇总见 `reverse_device_capture_summary.json`。下一步检查真实长序列的racecheck覆盖缺口。

### initcheck 对 TMA 写入的误报校准

最小五维张量对照已完成：TMA 和普通写入都将524288个元素写为7，
后续独立kernel加1，逐元素精确验证全部为8。生成代码确认TMA路径含fence、store完成等待。
普通写入initcheck退出0、0错误；TMA路径退出42，报告524288条未初始化读取。
因此当前工具链确实会对正确TMA写入产生这类误报，不能用此前QLA的h读取报告直接认定根因。
这并不证明QLA全部h访问正确。证据见 `tma_initcheck_control/summary.json`。
真实输入原版/fence的initcheck对照均命中同一h读取报告，随后主动停止，未完成数值测试；
部分结果见 `failure54_initcheck_partial_summary.json`。四组真实长序列racecheck则完整退出0，
排除已单独研究的kkt内核后均为0 hazards，见 `failure54_real_racecheck_summary.json`。

## 2026-10-06: named-barrier collision confirmed

Fresh installed prepare_h CUDA d408919516ef shows ID3 used by S threads0..127 and Y threads256..383. Original fresh synccheck:15200 errors. Replacing only Y ID3 with unused ID8 removes reported divergence but exposes896 Missing-wait reports at bar_3.arrive. Guarding that unused arrival by calc_mt lets the probe reach fused_chunk_gdr_fwd, which also reports divergence (rank0:15744 errors; rank3 see JSON).

TileLang engine runs ThreadSync(shared) then ThreadSync(shared.dyn). Each creates a new ThreadPartialSyncRewriter starting allocation at3 and preserves existing numbered calls. Captured IR before second pass shows Y assigned3. Diagnostic wrapper resets automatic IDs before second pass and reassigns all groups together: S3/X4/Y5. Both this compiler correction and prepare_h-only ID8 correction complete rank0/micro08 forward/backward with finite values. Initial compiler-wrapper attempt failed on GlobalVar.name in diagnostic code; fixed with Op type guard and rerun exit0.

Independent CUDA control with distinct group IDs: numerical assertion passes, synccheck0errors. A delayed-warp numeric control (1024blocks,512threads) sharing ID3 causes47872 wrong reads; different IDs cause0wrong reads. This confirms a real correctness hazard, not the original training failure causal link. All scripts/generated sources/logs in named_barrier*; production checkout/env unchanged.

Sequential launcher2026827 completed the numeric collision control and started four-rank FLA reference stress (128inputs x16 repeats) using compiler correction. No completed stress result yet. Next: check stress and run full synccheck with compiler correction plus calc_mt guard; then training-level verification.

## 2026-10-06: C++ repair, compiler-level numeric regression, and perturbation tests

Delay tests completed512 QLA calls each original/compiler, no nonfinites. Shared-memory NaN initialization completed512 calls each original/compiler, no nonfinites. See named_barrier_perturbation_summary.json; these negative tests do not establish cause of original large gradients.

Direct TileLang regression: static/dynamic shared buffers in disjoint128thread groups. Original automatic compiler output gives54432 wrong values /262144. Python pass reallocation gives0; C++ reserved-ID fix gives0. Initial test used CUDA reserved identifier static and failed compilation; renamed buffers and reran both.

Actual C++ source patch tilelang_named_barrier_ids.patch preserves existing numbered calls and reserves their IDs before allocating new ones; includes named-barrier capacity bound. Built isolated libthread_sync_reserved_ids.so, registry tl.transform.ThreadSyncReservedIds. Build required installed TVM custom logging flag and an overlay explicitly deleting TargetKindNode's already-invalid copy constructor for newer FFI copy-trait detection. No installed files changed. First compile and first runtime-load attempts failed; final compile and CPU-load pass. Manifest records hashes/build requirements.

Launcher3308107 waits for completed poison run, passes C++ minimal regression, currently runs cpp_wait rank0/micro08 full synccheck. If that and poison_compiler pass, launches run_train_reverse_cpp_barrier.sh,120updates, original reverse virtual32 HSDP2x2 schedule. Training source uses isolated C++ pass and guards unused bar_3 arrival; no per-kernel backward hooks. Retains checkpoint empty_cache workaround only at DCP boundaries. Training not yet verified started. Root cause of original failure and production integration still incomplete.

## 2026-10-06 06:18 UTC: C++ fixed training live

C++ reserved-ID library finalized with TVM_LOG_CUSTOMIZE=1, SHA256 e483988b10457fdaeb32b1401f8cadeefe57696e95301f209eefaee3f68aca3f. Do not overwrite/rebuild this library while current training maps it. C++ minimal compiler regression:0wrong /262144; original54432wrong. Actual C++ fix plus unused-arrival guard: rank0/micro08 synccheck0errors, full numeric forward/backward, exit0.

Training started through launcher3308107 at workspace gradient-debug/20261006T061550Z_reverse-cpp-barrier. Rank PIDs3466837..3466840 live as of06:18UTC; all4 install markers have same library hash. Monitor3524709 writes reverse_cpp_barrier_stats/heartbeat.json using PID+starttime rather than mutable command titles. Data/model initialization is ongoing; no optimizer update recorded yet. No other GPU experiment scheduled.

Delayed actual-input dq discrepancy: rank0/micro04 relative L2vsFLA original0.02116279 vs Python compiler-fix0.00812450; rank1/micro27 original0.01723416 vsfixed0.00509594. Gradients finite and maxima unchanged; scalar-only evidence, not elementwise proof or original1e23cause. See named_barrier_delay_dq_ranking.json. Further targeted tensor capture should follow training, without overlapping GPU experiments.

Goal incomplete: all raw/normalized scans done and reported; original gradient-norm overflow reproduced earlier; definite compiler correctness defect isolated and C++ repair experimentally verified. Its causal link to the historical32GPU failure and final production dependency integration remain unproven.

## 2026-10-06：固定输入下的 prepare_h 数值错误与隔离修复

对重建的 layer21 rank0/micro04 输入加入 warp 延迟，原版 QLA 的 `dq` 在重复调用间变化。
修复 TileLang barrier ID 分配及多余 arrival 后，这种变化仍未完全消失；因此不能把它全部归因于 ID 冲突。

使用相同输入，各做 128 次重复计算，分组插入上游 `fence_proxy_async` 后得到：

| prepare_h 的 fence | fused_bwd 的 fence | 发生 dq 变化的后续重复次数 |
| --- | --- | --- |
| 有 | 无 | 0 / 127 |
| 无 | 有 | 127 / 127 |
| 无，其余文件有 | 有 | 127 / 127 |
| 有，其余文件有 | 无 | 0 / 127 |

来源：`dq_fence_ablation_summary.json`。继续拆分 prepare_h 的 9 处 fence，
仅保留 X/Y 写入共享内存后的两处就能使该案例的 128 次重复完全一致。
仅保留 X 也可稳定，但保留其余 fence、只缺 X 同样稳定；这些实验不能证明 X 是唯一必要条件。
来源：`dq_prepare_fence_ablation_summary.json`、`dq_xy_fence_ablation_summary.json`。

随后从完整修复版本捕获固定的 `fused_gdr_h` 参数，独立重放该算子，排除后续反向内核的影响：

- 分块预处理调用的 `ht`：四组均稳定，结果逐元素相同。
- 反向所需的状态重算调用：无 fence 版本后续 127 次全部改变 `h`，最多改变 25,408 个元素，
  相对第一次的最大绝对差为 0.380859375。
- X/Y 两处 fence、prepare_h 全部 fence、完整 fence 三组：128 次均稳定，三组的 `h` 逐元素相同。
- 所有比较值有限。未比较 `calc_mt=False` 时未写入且不被使用的 `mt`，避免误判。

来源：`prepare_h_fixed_inputs/`、`prepare_h_isolation_summary.json`、
`prepare_h_isolation_comparison.json`。这是实际输入上的算子正确性证据；
输入来自成功的现场重放，并非历史故障瞬间的内核输入缓存，尚不能证明原始 1e23 梯度的因果关系。

仅含 barrier 修复的训练在第 32 步被平台自动回收容器打断，第 30 步 checkpoint 已完整保存。
平台事件明确记录资源利用率自动回收，并非训练日志中的梯度异常。
容器正常重启后，已从初始配置启动 120 步完整修复训练 `reverse-cpp-fence`：
实际 C++ barrier 修复、42 处上游 fence、源码中的 calc_mt arrival guard。
源码副本位于 `qla_corrected_trial/`，共享安装包和正式仓库代码未修改。
当前训练仍在初始化；结果以 `train_reverse_cpp_fence.log/.exit` 为准，不能视为训练验证已通过。
