# 5k 配方梯度异常诊断

已在 `yhtian-h200-5` 的四张 H200 上复现巨大梯度，并独立确认、修复了两个底层内核缺陷。
目前证据支持内核同步问题是重要线索；原 32 卡失败时未保存内核操作数，不能认定历史故障只有一个根因。

## 数据范围

扫描覆盖配方引用的全部 39 个机器人数据源、496114 个 episode、600428307 个时间点。
[原始字段范围](raw_ranges.csv) 包含极值 episode UUID；
[归一化范围](normalized_ranges.csv) 按数据集列出 state 和相对 action 的统计。
扫描检查数值字段及真实字段映射、unification、normalizer，不包括视频逐帧解码或 VLM 文本语义审核。
相对动作统计覆盖扫描定义的时间点/动作边界，不表示枚举了所有增强组合。

| 项目 | 结果 |
| --- | --- |
| 原始 state/action 非有限值 | 0 |
| 归一化 state 范围 | [-520.0152, 58815.9102] |
| state 任一维绝对值 >10 的时间点 | 408628 |
| state 任一维绝对值 >1000 的时间点 | 895 |
| 相对 action 归一化、裁剪前范围 | [-10066331, 10066329] |
| 模型 action 裁剪后范围 | [-1.5, 1.5] |

一个明确异常的 state：

- dataset：`coin_agilex_cobot_decoupled_magic_4cam_30fps__7a937cd691be`
- episode UUID：`7210db69-bbbd-5efd-b27c-f0cdf0f83483`
- 时间：14.3333 秒，原始索引 430，state 第 0 维归一化后约 58815.91。

原始数据本身存在极大数值；q01==q99 时以 float32 epsilon 替代分母也会放大偏离常量的值。
这些是数据质量问题，应核查原始 episode、字段单位和统计区间。本修复没有裁剪 state 或重算统计，
以免用数据变更掩盖内核问题。已做的极值样本对照没有复现原故障，因果关系仍未证明。

## 梯度复现与修复

历史 32 卡任务完成 87 次更新，在第 88 步裁剪梯度前失败。
四卡复现实验用每卡 8 条逻辑 loader 重放 32 卡数据顺序，完成 53 次更新，在第 54 步失败：
梯度元素仍全部有限，FP64 全局范数约 `9.245048884e23`，FP32 范数计算溢出。
巨大梯度集中于第 21 层线性 attention 反向边界。保存权重、batch 和 RNG 后的重放没有再次出现巨大梯度；
成功重放时捕获的操作数不能替代原失败现场的内核缓冲区。

独立实验证明：

1. TileLang 的 shared/shared.dyn 两轮 pass 重用了 named barrier ID，
   最小 CUDA 测试产生 54432 / 262144 个错误值；修复后为零。
2. QLA Hopper 缺少 async proxy fence。保存的真实输入在 warp 调度扰动下，
   127 次重复反向重算的 h 都与首次数值不同，最大差值 0.380859375；补齐 fence 后稳定。
   `prepare_h` 中未被使用的 barrier arrival 也已按消费者条件加上 `calc_mt` guard。

[补丁与构建说明](../../README.md) 包含依赖修复方式。
修复后同一四卡逻辑数据顺序完成 120 次更新，loss 和梯度范数全部有限，四个 rank 的范数完全一致。
第 54 步范数 17.1757，第 88 步范数 5.4598，最大范数 819.9566（第 40 步）。
第 30/60/90/120 步的主模型、优化器、EMA、loader/RNG checkpoint 均验证完整。
前 53 步的聚合数据/packing 统计与失败实验一致；这不是逐张量相等证明。
训练参数存储 FP32，计算 BF16，归约 FP32。

修复 wheel 另外通过了正常 `bc_veomni.cli` 四卡 HSDP 测试：4 次更新均有限，
第 2/4 步主模型、优化器、EMA 和额外状态文件完整，保存后继续训练成功，退出码 0。
该测试每卡累积 4 个 microbatch，未使用诊断 hook、编译器 pass 替换或缓存释放补丁。
它验证正常入口的可用性；120 步实验才负责覆盖之前复现的故障窗口。

最后将仓库构建脚本生成的 wheel 安装到训练 `.venv`，又通过了一次相同的正常入口测试：
4 次更新、第 2/4 步 checkpoint 和退出码 0 均已核验。最后一步 loss 为 16.7469，梯度范数为 122.7901。
安装后的版本为 `flash-qla==0.1.2+gerty1`、`tilelang==0.1.9+gerty1`；
`uv sync --frozen --offline --dry-run` 确认环境无需变更。

完整 `uv lock --check --offline` 受原有 `av` URL 元数据缓存缺失影响；
未修改的原始配置和锁文件也出现相同错误。新锁文件采用 uv 生成的两个本地 wheel 记录及元数据，
保留原依赖图，仅增加 Gerty 的直接 GPU 依赖；所有其他包的记录已逐项确认不变。
安装前后的依赖检查均只有项目已声明的 VeOmni/datasets 版本覆盖。

## 独立问题与证据位置

另一个 32 卡任务的最早已知错误是 `REDUCE_SCATTER` watchdog 超时，起因仍未确定。
诊断中还发现 checkpoint 的 CUDA 缓存占用可能导致通信分配失败；
120 步修复对照在保存前释放 CUDA 缓存。它与内核修复是不同改动，不能混为同一根因。

原始证据保存在共享仓库 `tmp/ai_tasks/20261004_gradient_diagnosis/`：

2026-10-09 存储清理后，下列证据中的训练 checkpoint 和大规模现场张量已删除；
报告、日志、JSON 审计、修复代码与编译产物继续保留，并保留四个代表性内核输入。
具体保留范围和复用入口见`gerty/tmp/ai_tasks/20261004_gradient_diagnosis/CLEANUP.md`。

- 全量扫描：`complete_range_summary.json`、`all_numeric/`、`all_states/`、`all_actions/`。
- 失败现场：`virtual32_hsdp_reverse/clip_failure_rank*.pt`。
- 内核测试：`prepare_h_isolation_summary.json`、`named_barrier_cpp_source_summary.json`。
- 120 步审计：`reverse_cpp_fence_final_audit.json`、`reverse_cpp_fence_trajectory_comparison.json`。
- wheel 测试：`wheel_regression/`、`wheel_prepare_h/`、`wheel_prepare_h_delayed/`。
- 正常 CLI 验证：`wheel_normal_entry.log`、`wheel_normal_entry.exit`、`wheel_normal_entry_audit.json`。
- 实际训练环境验证：`installed_wheel_normal_entry.log`、`installed_wheel_normal_entry_audit.json`。
- 完成审计：`final_evidence_audit.json`；原安装备份：`original_installed_attention.tar`。

短训练通过和固定输入测试只覆盖已测试的条件，不保证任意数据、调度和训练长度均无数值异常。
