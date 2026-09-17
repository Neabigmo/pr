<!-- REVIEW4-ROUTING-2026-09-05 -->
> **Review-4 routing notice (2026-09-05).** See [INDEX_REVIEW4.md](INDEX_REVIEW4.md) and the Review-4 runbook for updated contracts, known incompatibilities and outstanding integration gates. The original text below is preserved as historical context; it is not a claim that the new protocol or real experiments have passed.

# 给杨一横：当前 PR Mini-Pilot 项目大体情况

这份文档不是写给代码 agent 的，而是给你自己看的。目的只有一个：**把现在这个项目到底是什么、做到哪一步、哪里还不稳、下一步该怎么走，说清楚。**

---

# 一句话概括

现在这个项目已经不是一个“模型想法”，而是一个相当完整的 **Protein–RNA 联合序列设计 mini-pilot 框架**。

它已经具备：

- 明确的数据构建逻辑；
- 两个单分子结构先验；
- 一个跨分子 compatibility field；
- 分阶段训练；
- 严格 final-test 隔离；
- official baseline；
- internal controls；
- 解释性实验；
- robustness；
- 多 seed 统计；
- full-1000 final refit。

但是它**还不能被称为“实验已经准备完成”**。

第二轮重新审阅后，我又发现了几个之前没有抓出来的硬问题。好在这些问题主要集中在：

1. official baseline wrapper；
2. refit schedule；
3. 控制实验公平性；
4. interface 定义；
5. joint validation；
6. final evaluation 预算。

这些都可以在正式 GPU 训练前修好，而且**不会推翻核心模型思路**。

---

# 1. 我们现在到底在做什么

任务可以简单写成：

```text
给定 Protein backbone + RNA backbone
设计 Protein sequence + RNA sequence
```

或者做条件设计：

```text
RNA 已知 -> 设计 Protein
Protein 已知 -> 设计 RNA
```

核心思想一直没有变：

```text
最终序列选择
=
单链骨架本身允许什么
+
另一条分子在局部环境里更偏好什么
```

也就是：

```text
Structural Prior + Cross-molecular Selection
```

这条逻辑目前仍然是整个项目最清楚、也最有价值的地方。

---

# 2. 模型现在是什么样

## 2.1 Protein structural prior

Protein 单独学习：

```text
protein backbone -> amino-acid preference
```

不看 RNA。

Protein backbone 输入只使用 backbone geometry，不读 native side-chain identity。

---

## 2.2 RNA structural prior

RNA 单独学习：

```text
RNA sugar/phosphate backbone -> nucleotide preference
```

不看 Protein。

为了避免直接泄漏 native base identity，目前 RNA 结构输入不使用那些高度碱基特异的原子。

这会牺牲一部分 base-plane 信息，但换来了一个更干净的 inverse-folding 任务。

---

## 2.3 跨分子 DM-ICF

核心是：

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

可以把它非常直白地理解为：

### C

```text
一般来说，某个 amino acid 和某个 nucleotide 是否匹配
```

是一个全局 20×4 matrix。

它不是 PMI 初始化，而是小随机初始化后从复合物中学出来。

### DeltaC_ij

```text
在这个具体局部几何环境里，上面的全局匹配应该怎么修正
```

所以 C 是“通用规律”，DeltaC 是“局部修正”。

### alpha_ij

```text
一个位置附近有很多 partner 元素，到底谁更重要
```

不是一对一配对。

一个 Protein residue 可以综合多个 RNA neighbour，一个 nucleotide 也可以综合多个 Protein neighbour。

---

# 3. 数据是怎么设计的

mini-pilot 的目标数据量是：

```text
Protein prior       1000
RNA prior           1000
Protein-RNA complex 1100
```

其中 complex：

```text
1000 development
100 final test
```

Development 再内部切：

```text
900 train
100 validation
```

然后最后不是拿 900 训练的模型直接测，而是：

```text
900/100 只负责选 epoch/超参数
-> 从头重新训练
-> 完整 1000 development 全部进入 final refit
-> 最后才看 final100
```

这个统计流程是合理的，而且比“直接900训练、100测试”更符合你最开始说的“1000就是要真正训练进去”。

---

# 4. final100 为什么比较严格

final100 不是普通随机 100。

现在设计要求：

- Protein P30 不和 development 重叠；
- RNA R80 不重叠；
- Rfam family 不重叠；
- exact sequence 不重叠；
- test-linked family 也不能偷偷进入 Protein/RNA prior pool。

因此它更接近：

```text
strict bilateral OOD test
```

而不是 IID random split。

这会让指标比普通随机划分更难看，但科学价值更高。

---

# 5. 训练现在分几步

主模型仍然建议保留清晰的六阶段：

```text
P -> R -> C -> Delta -> Alpha -> Joint
```

## Stage P

训练 Protein structural prior。

## Stage R

训练 RNA structural prior。

## Stage C

冻结 P/R prior，只学全局 20×4 C。

## Stage Delta

冻结 C，学习：

```text
结构环境 -> DeltaC_ij
```

## Stage Alpha

学习邻域里谁更重要。

第二轮审阅后我建议把这一阶段再做得更干净：

```text
DeltaC 先冻结
只训练 alpha relevance
```

这样解释性更强。

## Stage Joint

最后才做小学习率联合协调。

但我现在建议：

```text
C 在 Joint 中继续冻结
```

不要再让全局 C 漂移。

这样最后画出来的 C 仍然是那个真正独立学到的 global anchor。

---

# 6. 现在已经做得比较好的地方

我认为下面这些已经比较成熟了。

## 6.1 数据泄漏意识很强

现在不是只检查 exact sequence。

已经考虑：

- P30；
- R80；
- Rfam；
- multi-chain partial overlap；
- final-test family 进入 pretraining pool；
- mother sample；
- modified residue canonicalization。

这比很多小规模算法论文的数据划分都要认真。

---

## 6.2 主方法与 baseline 的关系已经理顺

ProteinMPNN 和 NA-MPNN 很重要，但不能把它们写成和 DM-ICF 完全同条件。

原因很简单：

```text
ProteinMPNN 不看 RNA identity
NA-MPNN RNA reference 不看 Protein identity
DM-ICF 会看 partner
```

因此它们主要回答：

```text
在同一个 target backbone 上，标准单侧 inverse-folding 模型能做到什么程度？
```

真正证明“partner coupling 有用”的应该是内部控制：

- dual structural prior；
- + C；
- + DeltaC；
- + alpha；
- partner-blind；
- geometry-only capacity control；
- partner scramble；
- local partner mutation。

这个论文逻辑是对的。

---

## 6.3 final100 的实验维度很丰富

不只是 recovery。

已经设计了：

- Protein conditional；
- RNA conditional；
- joint；
- interface / non-interface；
- NLL；
- recovery；
- calibration；
- partner scramble；
- counterfactual mutation；
- C vs PMI；
- DeltaC；
- alpha；
- noise；
- edge removal；
- partner hiding；
- decode order；
- SPIR；
- data efficiency；
- multi-seed；
- paired bootstrap。

所以从论文完整性来说，实验框架已经不缺内容。

现在反而需要**收敛主线**，不能继续加实验名。

---

# 7. 第二轮审阅真正发现的问题

这是现在最重要的部分。

## 问题 1：official baseline runner 目前其实还不能放心直接跑

之前我们已经做了很多 baseline wrapper，但这次对照 pinned upstream 源码后，发现还有真实 CLI contract 问题。

ProteinMPNN runner 中存在：

- training data root 指错层级；
- 传入官方脚本不存在的参数；
- `mixed_precision` 参数形式不对；
- 官方脚本本身没有 seed 参数。

NA-MPNN 更明显：

它的 `na_run.py` 实际入口就是：

```text
python na_run.py config.json
```

不是一堆 `--xxx` 参数。

所以这部分必须先修。

这不是模型问题，是工程 wrapper 问题。

---

## 问题 2：NA-MPNN RNA 概率有一个 C/U 顺序风险

我们的 RNA alphabet 是：

```text
A U G C
```

但当前 exporter 的 NA-MPNN column 顺序存在：

```text
A C G U
```

这种错误会直接把 U/C 的概率解释反。

这是一个非常具体、也非常值得庆幸现在发现的 bug。

必须在正式 baseline evaluation 前修掉并加单测。

---

## 问题 3：final refit 虽然“用了1000”，但 schedule 被压缩了

这是我觉得第二轮最关键的训练问题之一。

Development 如果：

```text
max_epochs = 150
best epoch = 20
```

那 development 的第20个 epoch 其实只走到了整个 curriculum 的大约 13%。

但 refit 当前如果只训练20个 epoch，会把：

- curriculum；
- gradual unfreezing；
- cosine LR；
- joint task schedule；

全都压缩到这20个 epoch 里走完。

这等于 full1000 refit 用了另一个训练算法。

所以 refit 应该重放 development schedule 的“前20个epoch”，而不是把150 epoch schedule压缩成20。

这个一定要修。

---

## 问题 4：scratch baseline 现在不完全公平

Scratch joint 是随机初始化。

但它当前也走 gradual unfreezing。

也就是说一堆**随机初始化的 encoder layer 在前期还被冻结**。

对 pretrained model，gradual unfreezing 是合理的；

对 scratch model，这相当于绑着手和预训练模型比赛。

所以 scratch control 必须所有层从 step0 就能训练。

---

## 问题 5：interface 现在有两个定义

Screening 用的是：

```text
全重原子 6 Å contact
```

而 runtime graph 用的是：

```text
模型指定原子 + 8 Å + neighbour cap
```

现在训练/评估里的 interface mask 主要来自 runtime PR graph。

这会造成：

```text
“interface 指标”依赖我们自己的模型图怎么构
```

不够干净。

我建议今后明确分开：

```text
biological interface = full heavy atom 6 Å
model PR receptive field = 8 Å graph
```

前者负责：

- interface loss；
- interface recovery；
- external baseline interface mapping；

后者只是模型消息传递。

这会明显提高实验的可信度。

---

## 问题 6：Joint validation 还没有完全模拟真实 joint decoding

现在 joint validation 里有一个双方 full-mask 的一次 forward。

但是当 partner token 是 unknown 时，DM-ICF 本来就不给这个未知 partner contribution。

所以这项 metric 很大程度只在测 structural prior。

真正 joint inference 是：

```text
逐步生成
已经生成的位置变成 known
然后另一条链开始利用这些信息
```

因此 checkpoint selection 更合理的方式应该是：

```text
teacher-forced sequential pseudo-NLL
```

在 validation 上用固定 decoding order，逐 token 用 native token teacher forcing。

这样才和真实 mixed-order decoding 接近。

---

## 问题 7：final100 的 full suite 现在可能太重

现在如果：

```text
100 complexes
× 3 seeds
× 64 candidates
× 多种 SPIR
× 三种 decoding order
× 每个 token 一次 full forward
```

计算量会非常夸张。

可能训练没多久，最后 evaluation 自己跑几天。

所以我建议把实验分级：

### 所有3 seeds都跑

- core NLL/recovery；
- interface；
- joint pseudo-NLL；
- calibration；
- primary controls；
- seed stability。

### 最重的 robustness/mechanistic 分析

主 seed 跑完整。

64 candidates 只保留给真正主生成结果；一些 order/SPIR ablation 用16个即可。

这样不会降低论文质量，反而更规范。

---

## 问题 8：实验太多，不应该全部叫“primary”

我们现在有30多个测试，这是实验完整性的优点。

但论文里真正 primary hypothesis 最好只有4–5个。

建议主假设集中在：

1. Full DM-ICF > dual prior；
2. Full > partner-blind / geometry-only；
3. contextual field > C-only；
4. partner scramble 显著破坏 interface prediction；
5. alpha top-edge removal 比 matched lower-alpha edge 更重要（可选）。

Holm 只校正这几个。

其他大量实验作为 secondary / exploratory。

这样论文会从“做了很多实验”变成“主张非常集中，但支持证据很丰富”。

---

# 8. 现在项目处于什么阶段

我会这样给它打状态。

## 科学概念：绿

核心模型逻辑已经很稳定：

```text
structural prior + C + DeltaC + alpha
```

没有必要再大改思想。

---

## 模型核心代码：绿偏黄

主要模块都已经有。

现在需要的是：

- 更严格的 stage freeze；
- refit schedule 修正；
- joint validation 修正。

不是重写模型。

---

## 数据 pipeline：黄绿

流程已经相当完整：

```text
RCSB
-> screen
-> RNA chain view
-> MMseqs
-> Rfam
-> strict freeze
-> leakage audit
```

但真实1000/1000/1100数据还没有最终跑出来并冻结。

所以现在是：

```text
pipeline ready
!=
data ready
```

---

## official baseline：黄偏红

想法和转换器已经有，但执行 wrapper 还存在 pinned upstream CLI contract bug。

这是当前最明确必须修的工程点。

---

## final evaluation：黄

内容非常全，但计算预算过重，primary/secondary 还需要重新分级。

---

## 论文结果：红

因为现在还没有真正完成：

- frozen real dataset；
- GPU training；
- final100 evaluation。

所以现在不能谈“模型效果已经证明”。

---

# 9. 下一步不要再做什么

现在最不应该做的是：

- 再加新模块；
- 再加一个 fancy loss；
- 再加一个新的 attention；
- 再塞一个 LLM trick；
- 直接上 A800 长训练；
- 一边看 final100 一边调 SPIR。

现在项目的问题不是创新不够，而是：

```text
已经有很多东西，需要把合同彻底封死
```

---

# 10. 我建议接下来的实际顺序

## 第一阶段：把代码合同封死

先修：

1. baseline CLI；
2. NA-MPNN C/U mapping；
3. refit schedule；
4. scratch fairness；
5. C/Delta/alpha freezing；
6. canonical interface；
7. joint validation；
8. config dead keys。

然后跑一个极小 end-to-end smoke。

---

## 第二阶段：真正构建数据

开始：

```text
RCSB download
-> screening
-> RNA pool
-> joint clustering
-> Rfam
-> final100 first
-> prior purge
-> 1000/1000/1000 freeze
```

此时重点不是训练，而是先看：

- 最后能不能凑够；
- 长度；
- resolution；
- NMR比例；
- interface size；
- P30/R80/Rfam component size；
- strict test 是否过度偏移。

---

## 第三阶段：tiny real smoke

真实数据不要先全训。

先抽：

```text
20 Protein
20 RNA
20 complexes
```

跑完整 P→R→C→Delta→Alpha→Joint。

只确认：

- loss 会降；
- shape 对；
- freeze 对；
- checkpoint 对；
- refit 对；
- inference 能生成。

---

## 第四阶段：mini-pilot 正式训练

再上：

```text
1000 + 1000 + 1000
3 seeds
```

并同步 official baseline / internal controls。

---

## 第五阶段：final100 一次性评估

在所有配置冻结之后才打开 final100。

从这里开始：

```text
结果只能解释
不能再反向改模型
```

---

# 11. 如果最后结果不理想怎么办

这个 mini-pilot 最大的价值之一，就是现在规模小。

如果结果发现：

### Protein prior 好，RNA prior 差

说明 RNA backbone representation 需要加强，而不是 DM-ICF 本身一定错。

### C 有增益，DeltaC 没有

说明全局 compatibility 有信号，但 contextual residual 设计过强/数据不够。

### DeltaC 好，alpha 没增益

说明简单距离权重可能已经够用。

### partner-blind 和 full 差不多

说明模型没有真正利用 partner identity，这是最重要的负结果之一。

### conditional 好，joint 差

说明核心 interaction 学到了，但 simultaneous decoding/inference 机制需要重做。

所以这个阶段其实非常适合做诊断。

---

# 12. 我现在对这个项目的总体判断

我的判断比前一轮更谨慎，但并没有更悲观。

核心方法本身仍然是成立的，而且现在比最开始清楚很多：

```text
结构先验
+
全局 AA-base compatibility
+
局部 geometry-dependent correction
+
multi-neighbour relevance
```

真正的问题不在“这个想法是不是太简单”，而在于：

```text
能不能用非常干净的实验把四层信息拆开证明
```

现在的数据设计、stage training、internal controls、strict OOD test，已经基本围绕这个问题建立起来了。

第二轮审阅发现的这些 bug，恰恰说明现在做正式大训练还早了一步；但也说明**现在发现非常划算**，因为它们都属于“训练前可修”的问题，而不是跑完三天以后才发现结果不能解释。

所以目前最合适的项目定位是：

> **科学设计已经定型，工程实现接近收口，正式数据与实验尚未开始。**

不是重新设计模型的时候了。

接下来应该做的是：

```text
修合同
-> smoke
-> 冻结数据
-> 小规模真实训练
-> 正式 mini-pilot
```

如果这套 mini-pilot 能把 `dual prior -> C -> DeltaC -> alpha -> joint` 的增益链条和 partner-dependent evidence 跑出来，后面再扩到 PRI100K / predicted structures / specialized CRISPR fine-tuning，就会非常顺。
