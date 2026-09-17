# Fig.3 投影层消融实验 — 图表说明（配套 fig3_new.png / fig3_new.pdf）

> 图中只包含数据本身（柱状图、误差条、坐标轴与 X 轴标签），不含标题、图例、
> 显著性标记；实验设置、指标含义、统计检验、图注等全部文字说明见本文件。

---

## 1. 图中各元素含义

图中只含数据元素（无标题、无图例、无显著性标记）：

| 图中元素 | 含义 |
|---|---|
| **柱高（Y 轴）** | BLEU-1 指标值，为 **10 名被试的均值**（leave-one-subject-out，见下） |
| **误差条（error bar）** | 10 名被试间的 **±1 标准差（s.d.）** |
| **X 轴 4 个柱** | 4 种投影层结构（变体定义见第 3 节），绿色柱 **Joint Linear (Ours)** 为本文方法（身份由 X 轴标签标明） |
| **X 轴标签下 (n=10)** | 该柱汇总的被试数（每组均为 10） |

> 显著性检验结果（`**` / p 值）不在图中展示，见本文第 4 节与
> `statistical_tests.json`，论文正文/图注以文字形式引用。

---

## 2. 各项评测指标含义

所有指标均在**段落级测试集**上计算（生成文本 vs. 参考文本），数值越高越好。

| 指标 | 含义 | 取值范围 | 说明 |
|---|---|---|---|
| **BLEU-1** | **1-gram（单词级）精度**：生成文本中有多少比例的词出现在参考文本中 | 0–1 | 衡量**用词/内容重合度**，对短文本最稳健；本图主指标 |
| **BLEU-2** | 2-gram（相邻词对）精度 | 0–1 | 衡量局部搭配 |
| **BLEU-3** | 3-gram 精度 | 0–1 | 衡量短语级流畅度 |
| **BLEU-4** | 4-gram 精度 + 简短惩罚（brevity penalty） | 0–1 | 传统机器翻译主指标；EEG→文本任务上生成文本短、数值天然偏低（0.01–0.02 属正常区间） |
| **METEOR** | 基于词对齐的指标，**同时考虑精确率与召回率**（召回权重更高），并对碎片化（fragmentation）施加惩罚 | 0–1 | 与人类判断相关性通常优于 BLEU |
| **BERTScore-P** | 基于预训练语言模型**上下文词向量**的相似度：**精确率**（生成词中有多少能在参考中找到语义匹配） | ~0.5–0.7（中文） | 容忍同义改写，比 BLEU 更宽松 |
| **BERTScore-R** | 同上，**召回率**（参考内容有多少被生成覆盖） | ~0.5–0.7 | — |
| **BERTScore-F1** | 上述 P、R 的调和平均 | ~0.5–0.7 | BERTScore 的综合分 |

> 说明：BLEU-1 在 0.13–0.16、METEOR 在 0.08–0.09、BERTScore-F1 在 0.60 左右，
> 与论文正文报告的 ~0.13 BLEU-1 为同一合理区间（原版无效实验为 1e-5 量级）。

---

## 3. 实验设置（图中未显示的文字信息）

**比较对象（4 种投影结构，唯一变量 = token 生成拓扑）：**

| 变体（图例名） | 结构 | 投影参数量 |
|---|---|---|
| **Single Token** | 瓶颈 MLP（256→4352→768，隐层 ReLU+Dropout）+ 终端 tanh，输出**单个**向量复制 16 次作为条件 token | 4.46M（4,461,568） |
| **Joint Linear (Ours)** | 两层线性映射（256→358→12288，层间无激活）+ 终端 tanh，**一次性联合产出全部 16 个 token**；为全文模型 `eeg_to_bart`（Linear 256→12288 + Tanh，3.16M）的参数量对齐低秩实现 | 4.50M（4,503,398） |
| **Transformer** | 升维 MLP（256→960→768）复制 16 份 + 可学习位置编码 + 单层 Transformer encoder（d=768，8 头，FFN 1024）+ 终端 tanh | 4.94M（4,938,688） |
| **Independent Heads** | **16 个相互独立**的两层 MLP 头（各 256→272→768）+ 可学习位置偏置 + 终端 tanh | 4.49M（4,485,376） |

**公平性控制：**

- 4 种结构**终端激活统一为 tanh**，投影参数量对齐在 4.5M ±10% 以内；
- EEG 编码器**严格冻结**：加载 Stage-1 对比学习权重，全程 eval 模式 + `torch.no_grad()`，不接收梯度；
- 仅投影层与 BART LoRA 适配器（r=8，占解码器参数 0.31%）参与训练；
- 4 变体使用**完全相同**的数据子集、有效 batch size 128（32×4 梯度累积）、学习率 4e-4、cosine 调度 + 3 warm-up epochs、训练 **25 epochs**、AMP 混合精度；
- 数据：10 名被试（sub-04/05/06/07/08/09/10/13/14/15），段落级（3 行拼 1 段），每被试使用 35% 段落（4 变体同比例、同子集，保证消融公平）；
- 协议：**leave-one-subject-out（LOSO）**——留出被试的样本作验证/测试，其余被试训练，共 10 个折；
- 共 40 个训练 run（10 被试 × 4 变体 × seed 42），0 失败。

---

## 4. 统计检验

- 方法：**Wilcoxon signed-rank test（配对符号秩检验）**，非参数、不要求正态；
- 配对单位：**被试**（同一被试在 Ours 与对照变体下的得分一一配对，n=10 对）；
- 显著性阈值：`***` p<0.001，`**` p<0.01，`*` p<0.05，`n.s.` 不显著。

| 对比（Ours vs.） | BLEU-1（均值 ± s.d.） | p 值 | 胜/负被试数 | 结论 |
|---|---|---|---|---|
| Single Token | 0.155 ± 0.012 vs. 0.133 ± 0.012 | **0.002** `**` | 10 / 0 | Ours 显著更优 |
| Independent Heads | 0.155 ± 0.012 vs. 0.144 ± 0.008 | **0.004** `**` | 9 / 1 | Ours 显著更优 |
| Transformer | 0.155 ± 0.012 vs. 0.149 ± 0.014 | 0.375 `n.s.` | 6 / 4 | 均值更高，差异不显著 |

**结论：** 多 token 条件显著优于单 token 瓶颈（+0.022，10/0 全胜）；
且**联合线性映射显著优于逐 token 独立头**（+0.011，9/1）——联合变换天然保留
token 间相关性，而因子化的独立头会丢失这种相关；与 Transformer 性能相当。

---

## 5. 论文图注（可直接使用）

**英文：**

> **Fig. 3.** Ablation on projection architectures for EEG-to-text generation.
> Bars show BLEU-1 on paragraph-level test sets, averaged over 10 subjects
> (leave-one-subject-out); error bars denote ±1 s.d. across subjects (n=10).
> All variants share a terminal tanh, are parameter-matched within ±10%
> (~4.5M projection parameters), and are trained identically with the EEG
> encoder strictly frozen (only projection and BART LoRA adapters updated,
> 25 epochs). Joint Linear (Ours) significantly outperforms Single Token
> (p = 0.002) and Independent Heads (p = 0.004), and performs on par with
> Transformer (p = 0.375; paired Wilcoxon signed-rank tests).

**中文：**

> **图 3.** EEG→文本生成中投影层结构的消融实验。柱高为段落级测试集 BLEU-1
> 在 10 名被试上的均值（留一被试协议），误差条为被试间 ±1 标准差（n=10）。
> 四种结构终端激活统一为 tanh、参数量对齐（~4.5M，±10%），并在 EEG 编码器
> 严格冻结（仅训练投影层与 BART LoRA 适配器）、25 epochs 的相同条件下训练。
> 配对 Wilcoxon 符号秩检验表明：Joint Linear (Ours) 显著优于 Single Token
> （p = 0.002）与 Independent Heads（p = 0.004），与 Transformer 性能相当
> （p = 0.375）。

---

## 6. 配套文件

| 文件 | 内容 |
|---|---|
| `fig3_new.png` / `fig3_new.pdf` | 论文用图（300 dpi，无标题文字） |
| `summary.csv` | 40 个 run 的完整指标长表（BLEU-1~4、METEOR、BERTScore P/R/F1） |
| `summary_grouped.csv` | 按变体分组的均值/标准差汇总 |
| `statistical_tests.json` | 配对 Wilcoxon 检验完整结果（p 值、胜负数、配对被试列表） |
