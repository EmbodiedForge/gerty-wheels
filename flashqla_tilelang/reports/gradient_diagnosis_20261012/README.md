# 三项 attention 修复的独立归因实验

2026-10-12，在 `yhtian-h200-5` 上完成三个修复开关的 8 组合实验。
结论：**保留的真实输入在调度扰动下出现的 h 和梯度不稳定，可归因到缺少 async proxy fence；
尚未复现历史约 1e23 的巨大梯度，不能把这一局部归因等同于历史训练根因。**

## 实验边界与方法

### 两个包在训练中的作用

FlashQLA 是 Qwen gated delta rule 线性注意力的 GPU 计算内核库。
TileLang 提供编写这些内核的语言及编译器，将内核转换为 GPU 执行的代码。
在本项目中，Qwen3.5 主干的 GatedDeltaNet 层通过 VeOmni/FLA 后端调用 FlashQLA。
编译通常发生在内核首次使用时，后续训练复用编译结果；编译器错误因此会影响每次实际计算。
这组补丁针对线性注意力计算路径，不改变模型结构、loss 公式或优化器。

### 三个修复点

1. **C：编译器屏障编号。** `shared` 和 `shared.dyn` 分轮插入同步操作时，原编译器只按当前轮
   `barrier_id_map_.size()` 分配编号，忽略前一轮已经占用的编号，使不同线程组误用同一个屏障。
   补丁先收集已有编号，再选择未占用编号。
2. **F：异步内存访问顺序。** 线程将数据写入共享内存后，异步矩阵计算需要正确看到这些写入。
   补丁在 Hopper 内核中补充 42 处 `fence_proxy_async()`，建立两种内存访问机制之间的顺序保证；
   它不替代线程同步屏障。来源为上游
   [c4d7234](https://github.com/QwenLM/FlashQLA/commit/c4d72349d23ad66c506f2d509ebdbf17b39dfc4a)。
3. **G：barrier 的条件一致性。** `prepare_h` 的消费者仅在 `calc_mt` 为真时等待 `bar_3`，
   生产者原来却无条件 arrive。补丁使 arrival 同样受 `calc_mt` 限制。

这些问题可以在内存地址合法、输出 shape 正确的情况下算出错误数值，因此不一定触发 Python/CUDA
异常。线程调度变化还可能使同一输入有时正常、有时错误。构建目录中的 CMake/Cython 兼容补丁
只解决构建问题，不属于上述三个数值/同步修复。

### 对照设置

- C：TileLang 屏障编号修复；F：FlashQLA Hopper 的 42 处 async fence；G：`bar_3` 的 `calc_mt` guard。
- 从原安装备份恢复 `tilelang==0.1.9`、`flash-qla==0.1.2` 到容器 `/tmp/`，不改训练 `.venv`。
- C 使用已保留的独立 C++ pass；F 使用只补 fence 的 QLA 源码；G 单独修改生成 CUDA 中的 arrival。
  每个组合独立进程、独立 JIT cache，并保存实际生成的 CUDA 源码。
- 两份 `prepare_h_fixed_inputs/call{0,1}.pt`：正常调度、延迟调度各跑 8 组合，每输入重复 128 次。
- 两份 `failure54_layer21_inputs/rank0/micro{04,08}.pt`：延迟调度跑 8 组合，每输入重复 32 次前向/反向。
  使用保存的上游梯度计算 `dq/dk/dv/dg/dbeta`，并以禁用 QLA 路由的 FLA 实现作为独立数值对照。
- 延迟方式：在 partial barrier 前，对 warp index 模 4 等于 0 的 warp 插入 `__nanosleep(10000)`。
  它不修改输入值，但会改变执行时序与生成代码；不能将扰动结果视为原训练执行的逐指令重放。
- 输入来自此前成功重放时的捕获，不是历史失败内核内部的原始操作数。
  原大型失败现场和训练 checkpoint 已按先前清理操作删除，本次未重建完整训练。

## 8 组合结果

正常调度下，8 组合的两份 h/ht 输出各自重复稳定，且首轮输出与全修复版逐元素相同。
延迟调度下的结果如下。变化次数均相对该组合的第一次输出，第一次不计入分母。

| C | F | G | call1 的 h 变化次数 / 127 | micro04 的 dq 变化次数 / 31 | dq 相对 FLA 的最大 L2 误差 |
|---|---|---|---:|---:|---:|
| 0 | 0 | 0 | 127 | 31 | 1.9226% |
| 0 | 0 | 1 | 127 | 31 | 1.9168% |
| 0 | 1 | 0 | 0 | 0 | 0.4796% |
| 0 | 1 | 1 | 0 | 0 | 0.4796% |
| 1 | 0 | 0 | 127 | 31 | 1.7254% |
| 1 | 0 | 1 | 127 | 31 | 1.5727% |
| 1 | 1 | 0 | 0 | 0 | 0.4796% |
| 1 | 1 | 1 | 0 | 0 | 0.4796% |

不含 F 的四组，h 最大重复差值均为 `0.380859375`。micro08 的 dq 也呈现相同的稳定性分界。
含 F 的四组在两份输入上所有被检查的输出/梯度均重复稳定。
FLA 与 QLA 的算法、精度路径不同，残余相对误差不能单独判定为缺陷。
这里没有逐张量保存所有组合的梯度，不把相同误差指标表述为组合间逐元素一致。
所有完整算子实验的输出及梯度均有限，未出现历史巨大梯度。

这组对照表明：F 单独足以消除本次复现的不稳定；C、G 单独或联合不能消除它。
结论只覆盖所测输入和扰动，不证明 C、G 在其他输入或执行时序下不会影响训练。

## C 和 G 分别修复了什么

### C：可独立复现的编译器数值错误

复用维护中的 `test_attention_barriers.py` 的最小 GPU 内核，各运行 10 次：

- 原编译器：每次 262144 个输出中有 48192–53760 个错误值。
- 只启用 C：10 次错误数全部为 0。
- 真实 prepare_h 的生成 CUDA 也存在原版不同线程组复用编号的情况；C 将编号分离。

所以 C 是真实的正确性修复，而不只是推测。但本次真实输入的 h/梯度不稳定并不依赖该冲突，
不能据最小测试就把历史训练巨大梯度直接归因于 C。

### G：可独立消除的同步诊断错误

固定 C=1、F=0，仅切换 G，在两份 prepare_h 输入上运行 CUDA Compute Sanitizer synccheck：

- G=0：`Barrier error detected. Missing wait.`，1248 个错误，诊断退出码 77。
- G=1：两个输入完成，`ERROR SUMMARY: 0 errors`，退出码 0。

这证明 G 消除了所测路径的 barrier 使用错误；它没有消除本次重复数值抖动。
原版在普通执行中完成，而诊断器在发现同步错误后终止执行，不能把诊断终止当成原训练崩溃复现。

## 证据与复用

所有脚本和结果：`gerty/tmp/ai_tasks/20261012_attention_attribution/`。

- `probe.py`：独立 C/F/G 开关、h 重放、完整梯度和 FLA 对照。
- `summary.json`：各组合重复稳定性、数值范围和误差汇总。
- `cross_case.json`：h/ht 首轮输出相对正常调度全修复版的对照。
- `source_audit.json`、各组合的 `*.cu`：实际生成代码，编号和 guard 审计。
- `provenance.json`：代码提交、GPU/驱动、四份输入的 SHA256。
- `control_{0,1}.json`：原版/修复版编译器最小实验的逐次错误数。
- `synccheck_{100,101}.log` 和 `.exit`：G 的单变量诊断对照。
- 各组合 `manifest.json`、`results.json`、`progress.jsonl`：完整运行参数和逐次结果。

执行均在指定远端容器进行。本地仅编辑/读取文件和检查 diff。
初始化时的缺少 `libz3.so` 失败保留在 `import_failure/`；修正实验进程的动态库搜索路径后重跑。
并行调度调整时，重复启动尝试被输出目录的排他创建阻止，未重复运行同一个组合或覆盖结果 JSON；
部分外层日志/退出码受到该尝试影响，以完整 `results.json` 和逐次记录核验有效实验。
每个正常 h 组合应有 256 条记录，每个完整梯度组合应有 64 条记录。

下一步若要确认历史巨大梯度的根因，需要重新获得会失败的完整训练现场，捕获异常执行本身的
中间状态，并完成单变量反事实对照。此次没有把“未再出现巨大梯度”作为修复因果证明。

## 官方新版升级实测

同日在 `yhtian-h200-5` 安装并测试了 PyPI 官方 wheel：

| 包 | 原项目版本 | 升级候选 |
|---|---|---|
| FlashQLA | `0.1.2+gerty1` | `0.1.3` |
| TileLang | `0.1.9+gerty1` | `0.1.15` |
| apache-tvm-ffi | `0.1.9` | `0.1.12` |

官方 QLA 0.1.3 要求 TileLang >=0.1.12；TileLang 0.1.15 要求 apache-tvm-ffi >=0.1.11,<0.1.13。
VeOmni 0.1.12 则固定了这三项的旧版，因此候选 `pyproject.toml` 同时更新直接依赖、移除本地 wheel
source，并在已有 `tool.uv.override-dependencies` 中显式覆盖这三项。没有升级 VeOmni 或 PyTorch。

容器不能直连 PyPI。官方 wheel 通过共享目录准备，SHA256 与保存的 PyPI 元数据一致。
为保留其他正在使用 `.venv` 的进程，通过远端 `uv pip install --target` 安装到
`/tmp/gerty-attention-upgrade-site`，仅实验进程使用该目录；没有覆盖 `.venv` 的已安装包。
实验没有应用本项目的 C/F/G 补丁，仍使用 PyTorch `2.11.0+cu130`。

### 实际结果

| 检查 | 官方新版结果 |
|---|---|
| 两份 prepare_h 输入，正常调度，各 128 次 | 输出有限，重复差异为 0 |
| 两份 prepare_h 输入，延迟调度，各 128 次 | 输出有限，重复差异为 0 |
| 两份完整算子输入，延迟调度，各 32 次前向/反向 | output/dq/dk/dv/dg/dbeta 全部有限，重复差异为 0 |
| micro04 / micro08 的 dq 相对 FLA L2 误差 | 约 0.4582% / 0.5372% |
| 最小独立屏障内核 | **失败：CUDA illegal instruction，退出码 1** |
| 真实 prepare_h 的 synccheck | **失败：14496 个错误，退出码 77** |

synccheck 的首个诊断是 `Barrier error detected. Divergent thread(s) in block.`，位于
`tl::__sync_thread_partial`。最小内核生成代码中仍使用同一个编号 3，两个调用的参与线程数分别为
128 和 384。它证明这个官方编译器组合仍未通过该正确性测试，但不能仅凭错误类别断言所有失败
都与旧版缺陷完全同源。QLA 0.1.3 的 `bar_3.arrive()` 也仍没有本项目的 `calc_mt` guard。

新版将 partial barrier 从模板调用改成普通函数调用。最初沿用旧版注入表达式的 delay0 测试
实际没有插入延迟，因此不用于证明扰动下的稳定性。修正表达式后用新缓存、新输出目录重跑：
`h_000_delay0_verified_delay/` 和 `gradient_000_delay0_verified_delay/` 才是上表的延迟结果。
两份 h 的生成代码各确认有 18 处 `__nanosleep(10000)`。

编译完整反向时还出现 `dk_fragment` 初始化检查和 vectorized loop 降为 serial 的警告。
这里记录原始诊断，不把警告单独判定为已证实的计算错误。

### 采用决定与配置状态

**官方新版消除了本次复现的 fence 数值不稳定，但没有通过全部同步正确性检查，暂不替换默认组合。**
没有在失败的低层检查之上继续进行短训练，也没有证明历史巨大梯度的根因。

项目 `pyproject.toml` 已恢复到本次升级尝试前的内容；`uv.lock` 与升级前逐字节相同。
原先用户的依赖修改均保留。候选配置保存为
`gerty/tmp/ai_tasks/20261012_attention_upgrade/pyproject.candidate.toml`。
完整锁解析曾因不可联网的 PyTorch URL 元数据缺失而失败，即使提供匹配的已安装依赖元数据也未解决；
没有手工伪造一个通过解析的锁文件。隔离安装和 GPU 实测成功执行，不等于完整项目锁解析成功。

升级证据目录：`gerty/tmp/ai_tasks/20261012_attention_upgrade/`。
其中 `summary.json`、`source_audit.json`、`wheel_verification.json`、`provenance.json` 保存数值结果、
生成代码检查、官方 wheel 哈希和输入来源；`control.log`、`synccheck.log` 保存失败诊断。
复现入口为该目录的 `run_probes.sh`、`run_gradient.sh` 和修正延迟注入后的 `run_delayed.sh`。
脚本使用独立输出目录防止覆盖，重新执行时需使用新的实验目录或 tag。

## 旧版仅加 fence 的同步检查

在 `yhtian-h200-5` 进一步直接检查旧版 TileLang 0.1.9 + FlashQLA 0.1.2 仅加 F 的组合（010）。
实验通过进程内导入路径和补丁开关选择组合，没有修改当前训练环境。
真实 `prepare_h` 使用两份保留输入、正常调度，每份执行一次；synccheck 的错误退出码设为 77。

| 检查 | 旧版仅 F（010） | 修复对照 |
|---|---|---|
| 最小屏障内核，10 次，每次 262144 个元素 | 每次 50304–54528 个元素算错，退出码 1 | 此前 C 修复对照，10 次均为 0 个错误 |
| 真实 prepare_h 的 synccheck | 15648 次错误，退出码 77 | 本次全部修复（111）：0 次错误，退出码 0 |

010 的首个同步诊断为 `Barrier error detected. Divergent thread(s) in block.`，位于
`tl::__sync_thread_partial<3, 128>()`。15648 是诊断器累计报告次数，不代表独立缺陷数量，
也不能与官方新版的 14496 次直接比较缺陷严重程度。本次最小内核表现为算错，未观察到该测试
触发官方新版实验中的 `CUDA illegal instruction`。

因此，**旧版仅加 fence 能通过此前有限输入的数值稳定性检查，但不能通过同步正确性检查**。
输出稳定、有限且接近参考值，并不保证所有线程都正确参与屏障。
本次全修复对照通过所测 synccheck，也不能外推为所有训练形状和执行路径完全没有问题。

本次脚本、逐次错误数、日志和退出码保存在
`gerty/tmp/ai_tasks/20261012_fence_only_check/`，其中
`control_0.json`、`synccheck_010.log`、`synccheck_111.log` 为直接证据。
生成代码与导入路径记录位于前述 attribution 目录下的
`h_010_delay-1_fence_only_check/` 和 `h_111_delay-1_fence_only_check/`。
