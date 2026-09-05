# 给杨一横：PR Mini-Pilot 项目状态 V3

这份文档只回答三个问题：

1. **现在这个项目到底做成什么样了？**
2. **还有什么没有做？**
3. **下一步应该怎么做，什么时候才算真正完成？**

---

# 一句话结论

目前这个项目的 **科学设计和代码框架已经接近“可以进入真实数据构建与小规模执行”的状态**，但还不能说“实验已经完成”。

现在完成的是：

- 模型结构；
- 数据规则；
- 基线接口；
- 训练阶段；
- loss；
- final100 隔离；
- fairness controls；
- 多维评估；
- 统计假设；
- 训练/测试安全锁；
- 大量自动合同测试。

还没有完成的是：

- 真正下载并冻结 1000 Protein；
- 真正下载并冻结 1000 RNA；
- 真正冻结 1000 development + 100 final complexes；
- 真正跑 ProteinMPNN / NA-MPNN；
- 真正跑 DM-ICF 三个 seed；
- 真正得到 final100 指标。

所以现在最准确的说法是：

> **研究工程已经从“想法/半成品代码”进入“正式实验前的完整协议化框架”，下一步是用真实数据验证整个工程。**

---

# 1. 我们的模型现在已经很清楚了

核心仍然是：

```text
序列选择
=
自身骨架结构先验
+
partner 对局部选择的修正
```

Protein 和 RNA 先分别学习自己的 structural prior：

```text
Protein backbone -> AA preference
RNA backbone     -> A/U/G/C preference
```

然后复合物阶段学习：

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

其中：

- **C**：总体上 AA 与 base 的相容趋势；
- **DeltaC**：当前具体局部几何怎样改写这个总体趋势；
- **alpha**：附近多个 partner 元素里谁更重要。

这一条主线目前没有必要再改。

---

# 2. 这次第三轮审查真正改了什么

这次不是“代码格式优化”，而是又找出了几处会影响实验结论的问题。

## 2.1 把 biological interface 和模型 graph 彻底分开了

以前有一个潜在问题：

- 数据筛选说 interface 是 full-heavy-atom 6 A；
- 模型 runtime 又用 8 A + neighbour cap 构 PR graph；
- 如果直接拿 graph edge 定义 interface，那么我们的模型会部分决定“什么叫 interface”。

现在改成：

### 正式 biological interface

```text
full-heavy-atom PR contact <= 6 A
```

在筛选数据时就冻结每个 residue ID。

### 模型 message graph

```text
8 A + max-neighbour cap + rich geometry
```

只负责传信息。

两者不再混淆。

这对论文公平性很重要。

---

## 2.2 C / DeltaC / alpha 的训练 ownership 更干净了

现在 primary 模型严格是：

```text
Stage C
只训练 C

Stage Delta
只训练 q + DeltaC

Stage Alpha
只训练 alpha relevance + tau

Stage Joint
上下文模块和预训练 encoder 做协调
但 C 继续冻结
```

这样后面我们才能比较有底气地说：

```text
C       global anchor
DeltaC  context correction
alpha   neighbour relevance
```

而不是三个模块一起乱动、最后很难解释。

---

## 2.3 full1000 refit 的 schedule 修正了

以前虽然做了：

```text
900/100 -> 选 best epoch -> full1000 重训
```

但是有一个隐藏问题：

如果 best 是第20 epoch，而 development 原计划最多150 epoch，那么 refit 把整个 curriculum / unfreezing / cosine schedule 压缩到20 epoch里走完，实际上换了训练算法。

现在修成：

```text
best = 20
refit 仍只跑20 epoch
但 schedule progress 按 20/150 走
```

也就是**重放同一个 training schedule 的前20个 epoch**。

这个修正非常重要。

---

## 2.4 scratch 对照不再被“绑手”

真正 scratch model：

```text
random init
all parameters trainable from step0
```

不再用只适合 pretrained encoder 的 gradual unfreezing 去冻结随机层。

同时我们也意识到：

> dual-prior checkpoint 在还没有训练 Delta/alpha 时，Delta/alpha 也恰好是零。

所以不能只看“是不是零”自动判断 scratch。

现在 internal control 会显式指定：

```text
scratch_joint=False
```

真正 scratch 才指定：

```text
scratch_joint=True
```

避免误判。

---

## 2.5 partner-blind control 也重新修正了

之前想让 partner-blind 也经过：

```text
C -> Delta -> Alpha -> Joint
```

后来仔细一看，这是不合理的。

因为 partner-blind 强制：

```text
cross-molecular correction = 0
```

那 C/Delta/alpha 的 loss 根本不会依赖这些参数，硬跑这些阶段没有意义，甚至会出现无梯度。

现在 partner-blind 正确路径是：

```text
dual structural priors
-> complex Joint adaptation
但始终看不到 partner correction
```

geometry-only 才走完整 C/Delta/Alpha/Joint，因为它仍有跨分子 geometry，只是不知道 partner 的具体 AA/base identity。

这个对 H2 公平性非常关键。

---

# 3. Official baselines 现在是怎么处理的

## ProteinMPNN

锁定官方代码 commit。

Primary baseline：

```text
同一 frozen 1000 Protein
随机初始化
900/100 选择 epoch
full1000 从头 refit
```

不是用 published pretrained weight 冒充“小数据从头训练”。

Published checkpoint 可以后面作为 secondary reference 单独报告。

## NA-MPNN / MPNN-fixbb

同样：

```text
同一 frozen 1000 RNA
随机初始化
900/100 选择训练 pass
full1000 refit
```

这次还修了一个很具体的 RNA 概率顺序 bug：

项目内部：

```text
A U G C
```

NA shared-token 对应：

```text
DA DT DG DC
```

不能写成：

```text
DA DC DG DT
```

否则 U 和 C 会交换。

现在有专门单元测试锁住这一点。

---

# 4. final100 现在比以前安全得多

以前虽然说“final100 不参与调参”，但代码层仍有一些可能绕过的路径。

现在正式 workflow 变成：

```text
1. 所有主模型训练完成
2. full1000 refit 完成
3. internal controls 训练完成
4. 只用 development 做 DeltaC audit
5. 只用 complex_val 做 runtime profiling
6. 冻结 evaluation protocol
7. hash 锁死 config / final100 manifest / B-C-E-F checkpoints / control checkpoints
8. 才允许打开 final100
```

这个 lock 文件是：

```text
EVALUATION_PROTOCOL_LOCK.json
```

它不是普通记录，而是用 SHA256 锁：

- base config；
- final100 manifest；
- runtime budget；
- primary training-ready record；
- control training-ready record；
- B/C/E/F checkpoint；
- partner-blind / geometry-only checkpoint；
- analysis seed；
- candidate budget；
- H1-H4。

因此 final100 出结果以后不能再“换一个看起来更好的 checkpoint”。

---

# 5. final100 的实验现在怎么分层

我们之前设计了很多实验，优点是完整，缺点是太重。

现在分成两层。

## Tier A

所有 3 个 training seeds × 100 complexes。

主要做：

- Protein conditional；
- RNA conditional；
- interface/non-interface；
- teacher-forced joint pseudo-NLL；
- partner scramble。

这是主统计真正需要的东西。

## Tier B

只用预先指定：

```text
analysis_seed = 20260905
```

做重型实验：

- counterfactual mutation；
- DeltaC；
- alpha；
- C vs PMI；
- noise；
- edge removal；
- partner hiding；
- geometry permutation；
- decoding order；
- SPIR；
- candidate diversity；
- calibration；
- dataset shift。

这样实验仍然足够丰富，但不会无意义地重复三倍重型计算。

---

# 6. 我们现在只保留 4 个 primary hypotheses

这是这次很重要的收敛。

不是“30个实验都做 primary”。

## H1

Full DM-ICF vs dual structural priors。

看 canonical-interface normalized NLL。

## H2

Full DM-ICF 必须同时胜过：

- partner-blind；
- geometry-only。

证明收益不是单纯来自：

- complex-task fine-tuning；
- 多了一堆参数；
- 多看到了 partner geometry。

## H3

Contextual field：

```text
C + DeltaC + alpha
```

vs

```text
C only
```

检验 context dependence 是否真的有价值。

## H4

partner sequence composition-preserving scramble 后：

```text
interface NLL 是否变坏
```

这是 model-interventional partner dependence evidence。

不能写成生物学“因果证明”。

这 4 个才做 Holm correction。

其他大量实验都属于：

```text
secondary / exploratory / descriptive
```

这样统计和论文故事都更干净。

---

# 7. C 和 PMI 的解释现在更严谨了

Primary C：

```text
小随机初始化
完全不看 PMI
```

训练后再比较 empirical PMI。

而且 empirical PMI 不是从模型自己的 capped graph 统计，而是直接从真实实验结构 full-heavy-atom contacts 统计。

还可以拆：

```text
base-facing
sugar-facing
phosphate-facing
```

另外我们增加了一个很重要的 **DeltaC population drift audit**：

```text
mean DeltaC
RMS DeltaC
mean-shift ratio
alpha-weighted mean
C_eff = C + mean DeltaC
```

如果 DeltaC 平均值很大，我们不会硬说：

```text
raw C = final population average
```

而会更诚实地说：

```text
raw C = stage-1 global anchor
C_eff = final effective global tendency
```

这会让论文解释更稳。

---

# 8. 数据下载/筛选已经比以前完整很多

现在包含：

```text
RCSB broad discovery
-> concurrent deterministic download
-> retry/backoff
-> .part atomic write
-> checksum
-> failures.tsv
-> coordinate QC
-> modified residue canonicalization
-> ribosome/spliceosome exclusion
-> full-heavy-atom canonical interface
-> joint MMseqs clustering
-> Rfam cmscan
-> final100-first freeze
-> prior-pool purge
-> post-freeze local-similarity audit
```

Rfam 现在还会保存：

- requested URL；
- resolved URL；
- download time；
- bytes；
- SHA256。

因此 reproducibility 比前几版强很多。

---

# 9. 现在我认为“已经做好的”部分

## 绿色：科学结构

- structural prior + interaction correction 主线；
- C + DeltaC + alpha；
- rich PR geometry；
- P/R 分开预训练；
- C/Delta/alpha 分阶段；
- balanced loss；
- random masking + patch + wrong-token；
- 0.10 A noise；
- edge dropout；
- joint decoding + SPIR。

## 绿色：公平性协议

- final100 first；
- P30/R80/Rfam/exact leakage；
- final100 purge from priors；
- canonical interface vs PR graph 解耦；
- official baseline immutable SHAs；
- 900/100 -> full1000 refit；
- refit schedule prefix；
- partner-blind / geometry-only；
- B/C/E/F checkpoint lock。

## 绿色：统计框架

- complex-level unit；
- 3 primary training seeds；
- H1-H4 only；
- Holm only H1-H4；
- secondary/exploratory separate；
- paired bootstrap / directional paired tests。

---

# 10. 现在仍然没有“做好的”部分

这一点必须说得非常清楚。

## 10.1 真实数据还没真正冻结

代码会构建：

```text
1000 Protein
1000 RNA
1000 development complex
100 final complex
```

但现在 repository 中还没有这套正式 manifest 成果。

## 10.2 正式 GPU 训练还没开始

所以现在没有：

- ProteinMPNN pilot 结果；
- NA-MPNN pilot 结果；
- DM-ICF P/R/C/Delta/alpha/joint learning curve；
- H1-H4 p-value；
- C-PMI correlation；
- SPIR gain。

## 10.3 CI 仍需要针对 Review3 branch 真正跑一次

代码已经增加大量测试，但最终是否“框架完成”，必须由最新 branch 的：

```text
compileall
pytest
config usage audit
Ruff correctness
CLI smoke
```

共同确认。

不能拿旧 main 的绿灯替代最新 review3 branch。

---

# 11. 下一步最合理的顺序

我建议你不要再继续“设计模型”。

下一步应该是：

### 第一步

让 Review3 branch 的 CI 真正跑绿。

### 第二步

合并到 main，冻结一个 code SHA。

### 第三步

真实下载和筛选数据。

### 第四步

先只做数据审计，不急着 GPU。

### 第五步

prepare official baselines，跑 CPU preflight。

### 第六步

极小 end-to-end smoke：

```text
2 Protein
2 RNA
2 complex
```

验证 P->R->C->Delta->Alpha->Joint->refit->eval 真的能串起来。

### 第七步

开始正式 1000/1000/1000 pilot GPU training。

### 第八步

development-only runtime profile。

### 第九步

freeze `EVALUATION_PROTOCOL_LOCK.json`。

### 第十步

一次性打开 final100。

---

# 12. 什么时候才叫“项目完成”

不是代码写完就叫完成。

至少要满足：

```text
真实数据 frozen
+ data audit PASS
+ official baseline PASS
+ DM-ICF 3 seeds full1000 refit
+ controls full1000 refit
+ evaluation protocol locked
+ final100 Tier A/Tier B done
+ H1-H4 statistics done
+ external baseline table done
+ all artifacts/checksums/env archived
```

到那个时候我们才能说：

> **这个 mini-pilot 实验完整完成，可以据结果决定是否进入大规模训练。**

---

# 我的当前判断

如果把状态分成：

```text
想法 -> 原型 -> 工程框架 -> 真实 pilot -> 大规模实验 -> 论文结果
```

我们现在处于：

```text
工程框架后期
```

已经非常接近：

```text
真实 pilot
```

核心模型不建议再大改。

现在最重要的不是“再想一个更高级的 trick”，而是把这套已经相当严密的 protocol **用真实数据跑通，并让结果来告诉我们哪里值得继续扩大**。
