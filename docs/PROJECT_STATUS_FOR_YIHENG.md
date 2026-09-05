# 给杨一横：PR mini-pilot 当前项目情况（v3 深度复查后）

## 一句话结论

现在这个项目的**模型设计已经基本定型，代码框架也已经接近真正可执行**；但我仍然不建议立刻开正式 GPU 训练。原因不是模型还没想清楚，而是科学实验里最容易被忽视的几件事——baseline 接口、refit 训练语义、测试界面定义、验证集泄漏和最终统计——刚刚经过第三轮深度审查并做了实质修复。

现在最合适的状态是：

```text
模型方案：      基本冻结
主代码：        v3 已系统修正
数据下载/筛选： 代码已具备，但真实 1000+1000+1100 尚未冻结
正式训练：      还没开始
下一步：        跑真实数据管线 + baseline preflight + 小样本 smoke test
```

---

## 1. 我们到底要做什么

这个小实验不是为了直接做到最终论文规模，而是用尽量小的数据把整个研究逻辑从头到尾走通：

- 随机 1000 个 Protein 结构；
- 随机 1000 个 RNA 结构；
- 1100 个真实 Protein–RNA experimental complexes；
- 其中 1000 个用于 development/最终 refit，100 个作为一次性 final holdout。

Protein 侧用官方 ProteinMPNN 从头训练作为参考；RNA 侧用官方 NA-MPNN / MPNN-fixbb 路线从头训练作为参考；我们自己的模型则从 Protein prior、RNA prior 一直训练到 DM-ICF 和 joint design。

这个 pilot 真正想回答的不是简单的“recovery 高不高”，而是：

> **结构先验之外，模型是否真的学到了 Protein 与 RNA 之间的元素选择规律？**

---

## 2. 我们的模型现在是什么

核心仍然很简单：

```text
序列偏好 = 自身结构先验 + 跨分子局部选择
```

跨分子部分：

```text
Gamma_ij = alpha_ij ( C + DeltaC_ij )
```

其中：

- `C`：全局 20×4 AA–base compatibility anchor，从小随机值开始学，不用 PMI 初始化；
- `DeltaC_ij`：这个具体局部几何环境对全局规律的修正；
- `alpha_ij`：周围多个 partner 中谁更重要。

这三个部分现在在训练阶段被刻意拆开：

```text
C 阶段      只学 C
DeltaC 阶段 只学 q + DeltaC
Alpha 阶段  只学 alpha/tau
Joint 阶段  C 固定，其余做轻量协调
```

这样以后画 C、DeltaC、alpha 的图才有解释意义。

---

## 3. 这次第三轮复查真正修掉了什么

前两轮看上去代码已经很完整，但第三轮从官方 upstream 源码和实验定义反推后，还是找出了几处如果直接跑会影响结论的问题。

### 官方 baseline 接口之前并不可靠

之前 ProteinMPNN 和 NA-MPNN 的 wrapper 有一些参数是假设出来的，并不完全对应 pinned upstream 的真实入口。现在已经按官方代码重新对齐，而且新增了专门 preflight。

尤其 NA-MPNN 还发现了一个很隐蔽的碱基顺序问题：我们的项目顺序是 `AUGC`，shared token 应当读成：

```text
A U G C
DA DT DG DC
```

旧逻辑可能把 C/U 对调，现在已经修正。

### 最终 1000 refit 以前会改变训练过程

以前如果 development 在第 12 epoch 最好，full-1000 refit 会把原本 150 epoch 的 curriculum/cosine schedule 压缩成 12 epoch，这其实已经不是同一种训练。

现在改成：

> best epoch 表示“原 schedule 的前 K 个 epoch”，full1000 只是在更多数据上重新走同样的前 K 段。

### scratch 对照以前不公平

预训练模型 gradual unfreeze 很合理，但随机初始化的 scratch 模型如果也冻结随机 encoder，就会天然吃亏。现在 scratch 从 step 0 就全部可训练。

### interface 定义现在真正分清了

以前模型的 PR graph 是 8 Å + neighbour cap，而筛选界面是 heavy-atom 6 Å。如果用 PR graph 来定义 interface recovery，那么改个 graph cutoff，测试集的“interface”也跟着变，明显不合理。

现在：

- 报告和 loss 的 interface = clean heavy-atom 6 Å；
- 模型 message graph = sequence-neutral 8 Å/capped graph；
- SPIR 在实际设计时只使用后者，避免偷看 native side-chain/base contact。

### prior validation 也变严格了

Protein 1000 里的 900/100 不再简单随机，而是 P30 family-disjoint；RNA 900/100 用 R80/Rfam component-disjoint。

这样选 prior checkpoint 时，不会因为 validation 里都是很相似的同家族结构而显得特别好。

### RNA 从 complex 抽链的 fallback 也进一步防泄漏

我们允许从其他 experimental Protein–RNA complexes 抽 RNA chain 做 RNA structural prior，这是合理的；但现在会明确排除**那 1100 个最终下游 complex 本身**的抽链视图，防止 prior 先见到后面要训练/测试的 exact RNA backbone。

---

## 4. 为什么现在还不能直接开 GPU

代码正确和数据正确是两回事。

现在还缺的不是“继续写模型”，而是真正完成：

```text
RCSB 下载
-> coordinate QC
-> MMseqs2 聚类
-> Rfam 注释
-> strict split
-> 1000/1000/1100 冻结
-> leakage audit
-> baseline converter smoke test
```

这一步跑完之前，任何训练结果都没有意义。

所以当前 GPU 状态仍然是：

```text
NO-GO
```

等上述数据和 baseline preflight 全绿后，就可以正式开始。

---

## 5. 训练时到底会怎么跑

Development：

```text
Protein prior: 900 train / 100 val
RNA prior:     900 train / 100 val
complex:       900 train / 100 val
```

它们只负责选择 epoch / checkpoint / 已预注册超参。

随后从头重训：

```text
Protein prior: 1000
RNA prior:     1000
complex:       1000
```

最后才打开那 100 个 final complexes。

所以你的原要求“最终确实用 1000 Protein + 1000 RNA + 1000 complex 训练”现在真正满足，同时又不会失去 validation。

---

## 6. 最终 100 会怎么测

最后不会只有 recovery。

核心一定包括：

- Protein conditional NLL / recovery；
- RNA conditional NLL / recovery；
- interface / non-interface；
- joint sequential design；
- partner scramble；
- partner mutation / local KL response；
- C 与 independent heavy-atom PMI；
- DeltaC 几何依赖和平均漂移；
- alpha 的 neighbour importance；
- noise / edge removal / partner hiding robustness；
- Protein-first / RNA-first / mixed order；
- SPIR；
- calibration；
- candidate diversity；
- 10/25/50/100% data efficiency；
- 三个 seed 的稳定性。

但第三轮复查后，我把这些实验分了级，不再把 30 多个东西都叫“primary hypothesis”。

真正 confirmatory 的只留一小组，写在 `configs/hypotheses.yaml`；其余作为 secondary/exploratory。这样论文会清爽很多，统计也更合理。

---

## 7. ProteinMPNN / NA-MPNN 应该怎么理解

这两个模型很重要，但它们不是和 DM-ICF 完全同输入的 competitor：

- ProteinMPNN 看 Protein backbone，不看 RNA identity；
- NA-MPNN fixed-backbone reference 看 RNA，不看 Protein identity；
- DM-ICF 的 conditional task 明确利用 partner。

所以它们回答的是：

> 在同一批单体数据和骨架上，一个成熟的 one-sided inverse-folding model 能做到什么程度？

真正证明“partner coupling 有价值”的证据来自我们内部的：

```text
prior only
+ C
+ DeltaC
+ alpha
partner-blind
geometry-only
partner scramble
counterfactual mutation
```

这个区分现在已经写入代码和文档，避免以后论文比较被审稿人抓住“不公平”。

---

## 8. 还有什么我刻意没有继续加

这次复查以后我反而更不建议继续往模型上堆 trick。

这个 pilot 暂时不加入：

- predicted structure augmentation；
- family-aware replacement sampling；
- dynamic token packing；
- automatic PCGrad；
- alternative assembly enumeration。

这些东西在大规模版本里可以做，现在加入只会让一个 1000-sample 机制验证变得难解释。

还有一个明确保留的限制：当前 loss 是 Protein/RNA、interface/non-interface balance，但没有再让每一条 individual chain 完全等权。因此论文里不能写“per-chain balanced”。小实验先把 single-chain vs multi-chain 分层汇报；如果未来正式大数据中多链样本很多，再设计真正的 chain-balanced loss。

---

## 9. 我认为这个项目现在是否合适

我现在对它的判断比前两轮更有信心，因为这次重点不是“再加几个功能”，而是把几个会让论文站不住的细节拆掉了。

目前最大的优点有三个：

1. **主假设很清楚**：结构先验 + partner-induced correction；
2. **核心模块可解释且可消融**：C / DeltaC / alpha 各自有明确职责；
3. **pilot 已经从“模型 demo”升级成一套可审计的实验设计**：数据、baseline、refit、final holdout、统计和解释性验证都有明确边界。

真正的风险已经不是“方案逻辑有大漏洞”，而是接下来真实数据筛选后，数据量、RNA 类型分布、multi-chain 比例以及实际训练效果是否支持我们的假设。

所以现在不应该继续无休止地改结构。下一步就是把真实数据冻出来，让实验说话。
