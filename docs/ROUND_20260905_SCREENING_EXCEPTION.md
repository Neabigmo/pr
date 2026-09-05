# 2026-09-05 本轮结构筛选临时策略

## 目的与范围

本文档记录本轮运行的临时筛选策略。它只适用于本轮数据处理，不修改正式
`pilot_v1` 协议，也不表示最终科学实验可以永久放宽质量控制。

本轮使用独立配置：

`configs/pilot_round_20260905_screening.yaml`

冻结阶段使用配套的本轮配置：

`configs/pilot_round_20260905.yaml`

正式配置 `configs/pilot.yaml` 仍保留原有分辨率/实验方法约束和原长度约束。

## 本轮明确变更

### 暂时跳过的两个筛选项

本轮不根据以下两个条件拒绝结构：

1. 结构分辨率是否不超过 4.0 Å；
2. 在没有可用分辨率时，实验方法字段是否包含 NMR。

程序仍然读取并记录 `method` 与 `resolution` 字段，但这两个字段本轮不参与
eligible/rejected 判定。其他结构检查保持启用，包括：

- DNA/核酸混合物排除；
- 不支持的修饰聚合物排除；
- Protein/RNA 链类型和长度检查；
- Protein–RNA 实际重原子接触至少 3 对，接触阈值 6 Å；
- 选中界面链的参考原子完整性；
- 复合物总 token 不超过 1,000；
- 界面缺失原子比例不超过 10%；
- 大型 ribosome/spliceosome 关键词排除。

### 长度约束

本轮使用以下长度范围，均为闭区间：

| 对象 | 最小长度 | 最大长度 |
|---|---:|---:|
| RNA | 10 | 500 |
| Protein | 40 | 2,000 |

对于复合物，Protein/RNA 参与界面的选中链仍受相应最小长度约束，并继续受
总 token 上限和界面质量约束；复合物的实际筛选逻辑以代码为准。

## 可复现命令

在 `pytorch-clean` 环境、代理 `127.0.0.1:7897` 下执行：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:PYTHONPATH = "src"

python -m pr_pilot.cli screen --kind protein `
  --config configs/pilot_round_20260905_screening.yaml `
  --download-manifest <workflow>/data/raw/protein/download_manifest.tsv `
  --out <workflow>/data/screened_round/protein

python -m pr_pilot.cli screen --kind rna `
  --config configs/pilot_round_20260905_screening.yaml `
  --download-manifest <workflow>/data/raw/rna/download_manifest.tsv `
  --out <workflow>/data/screened_round/rna

python -m pr_pilot.cli screen --kind complex `
  --config configs/pilot_round_20260905_screening.yaml `
  --download-manifest <workflow>/data/raw/complex/download_manifest.tsv `
  --out <workflow>/data/screened_round/complex
```

其中 `<workflow>` 是本轮工作流输出根目录，不应把生成的数据提交进代码仓库。

## 候选池补充与实际规模

首轮复杂体下载并筛选了 6,000 条，得到 1,043 条合格记录。为满足严格
测试集优先冻结的要求，随后从同一份 RCSB 发现结果中补下载了其余 292 条，
新增 52 条合格记录。两批 PDB ID 无重叠，合并后覆盖全部 6,292 条已发现候选，
复杂体合格总数为 1,095 条。

因此本轮数据实际支持的冻结规模是：复杂体开发集 1,000 条、最终测试集 95 条，
其中开发集仍严格拆为训练 900 条和验证 100 条；Protein/RNA 结构先验池仍按
1,000（900/100）冻结。正式 `pilot_v1` 的 1,100/100 目标不被本轮临时规模
覆盖，也没有用重复样本、预测结构或放宽其他生物学/界面条件来补足缺口。

## 注释复核

最终注释输出使用 `data/annotated_round_fixed2`。此前的
`data/annotated_round` 和 `data/annotated_round_fixed` 只作为审计留痕保留，
不得作为后续输入：首轮 Infernal `--fmt 2` 解析曾把目标列误当作查询列，导致
Rfam 标签异常偏少；代码已修复为读取查询名称列，并在合并全部 1,095 个复杂体
后完成全量重注释。

## 逐条进度与日志

每处理完一条 manifest 记录，筛选器立即向 `*_progress.jsonl` 写入一条
`record_complete` 事件并 flush。事件包含序号、总数、PDB ID、路径、状态、拒绝
原因、单条耗时和 UTC 完成时间；异常记录写入 `record_error` 后原样抛出。

复合物 16 分片调度器关闭子进程自己的进度条，使用各分片的 JSONL 日志汇总显示
一个实时总进度条；每个分片同时保留 `worker.log`。因此可以在不中断任务的情况下
检查：

- 总进度条：当前终端输出；
- 单条结构进度：`data/screened_round/complex_shards_16_logged/shard_*/progress.jsonl`；
- 分片标准输出和错误：对应目录下的 `worker.log`；
- 只有全部记录均完成后，才会写入合并后的 `complex_eligible.tsv`、
  `complex_rejected.tsv` 和汇总 JSON。

补充候选使用同样的 16 分片调度器，输出位于
`data/screened_round/complex_expansion_16_logged`；合并结果位于
`data/screened_round/complex_expansion`。首轮和补充批次分别保留原始逐条日志，
合并前已核验无 PDB ID 重叠且 6,292 条记录无遗漏。

## 风险与后续要求

放宽分辨率/方法筛选会增加低质量或元数据不完整结构进入候选池的可能性。因此，
本轮输出不能直接宣称满足正式 `pilot_v1` 的严格质量协议。若后续要用于正式
训练、最终测试或论文结果，必须重新启用正式筛选，或新建一个经过审查和版本化
的协议，并重新完成注释、泄漏审计、冻结和验证。
