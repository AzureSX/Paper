# 1 INTRODUCTION

**Keywords**

- self-supervised + recommendation
- contrastive + recommendation
- augmentation + recommendation
- pre-training + recommendation



# 2 DEFINITION AND TAXONOMY

![image-20230706093016451](E:\Paper\Review\自监督\images\image-20230706093016451.png)

## 2.1 Preliminaries

- 图

  我们让 $${\mathcal G}\,=\,({\cal V},\,{\cal E}) $$ 表示用户项目二部图，其中 $\cal V$ 是节点集（即用户 $\cal U$ 和项目 $\cal I$ ），$\cal E$ 是边集（即交互）

  图结构用邻接矩阵 $\mathbf{A}$ 表示，其中 $\mathbf{A}_{u i}=1$ 表示节点 $\cal u$ 和节点 $\cal i$ 相连

- 序列

  我们让 $\mathcal{I}\ =\ [i_{1},i_{2},...,i_{n}] $ 表示项目集。每个用户的行为通常被建模为有序序列 $S_{u}\,=\,[z_{1}^{u},z_{2}^{u},\ldots,z_{k}^{u}],(1\,\le\,k\,\le\,n) $, 且 $S\,=\,\left\{S_{1},S_{2},...,S_{m}\right\} $ 指整个数据集。在某些情况下，用户和项目与其属性相关联。我们使用 $\mathbf{X}\ =\ [\mathbf{x}_{1},\mathbf{x}_{2},...,\mathbf{x}_{m+n}] $ 来表示属性矩阵，其中 $\mathbf{X}\ =\ \mathbb{R}^{t} $ 是表示对象 $i$ 属性的多热向量

推荐模型的一般目的是学习用户和项目的质量表示 
$$
\mathbf{H}\in~~{{{\mathbb{R}}}}^{(m+n)\times d} = \left[{\mathbf{U}},\mathbf{V}\right]
$$



## 2.2 Definition and Formulation

SSL 提供了一种解决推荐中数据稀疏问题的新方法。然而，目前SSR还没有正式的定义。为了给后续该领域的研究打下坚实的基础，通过查阅收集的文献，提出了一个清晰准确的SSR定义，其三个关键特征概括如下

1. 半自动地利用原始数据本身来获取更多的监督信号
2. 结合自监督任务来使用增强数据（预）训练推荐模型
3. 自监督任务旨在提高推荐性能，而不是最终目标



考虑到推荐系统中不同的数据类型和优化目标，需要一个与模型无关的框架来制定 SSR

大多数现有方法可以概括为 **Encoder + Projection-Head** 架构，为了适应不同的数据模式，例如图、序列和分类特征，可以使用一系列神经网络，例如图神经网络（GNN）、transformers 和多层感知器（MLP）用作编码器 $f_{\theta}$，而投影头 $g_{\phi}$（在生成模型中也称为解码器）通常是轻量级结构，例如线性变换、浅层 MLP 或非参数映射。编码器 $f_{\theta}$ 旨在学习用户和项目的分布式表示 $\mathbf{H}$，而投影头 $g_{\phi}$ 则针对推荐任务或特定的自监督任务细化 $\mathbf{H}$，基于该架构，SSR可以表述如下：
$$
f_{\theta^{\ast}},g_{\phi^{\ast}},\mathbf{H}^{\ast}=\mathop{\arg\min}\limits_{{{f_{\theta}\,,g_{\phi}}}}\,{\cal L}\left(g_{\phi}\big(f_{\theta}\big({\cal D},\tilde{\cal D}\big)\big)\right)
$$
其中 $\cal D$ 表示原始数据，$\tilde{\cal D}$ 指满足 $\tilde{\cal D}\sim {\cal T(D)}$ 的增强数据，${\mathcal{T}}(\cdot) $表示增强模块，${\cal L}$ 是合并损失函数，可分为推荐损失任务 ${\cal L}_{rec}$ 和借口任务 ${\cal L}_{ssl}$ 的丢失。通过最小化方程(1)，可以学习最优编码器 $f_{\theta^{\ast}}$ 、投影头 $g_{\phi^{\ast}}$ 和表示 $\mathbf{H}^{\ast}$ 来生成质量推荐结果



## 2.3 Taxonomy

SSR 与其他推荐范式的区别在于强调自我监督任务在其方法中的作用。我们根据自我监督任务的性质将现有的 SSR 模型分为四类：对比型、预测型、生成型和混合型

### 2.3.1 Contrastive Methods

![image-20230706104955282](E:\Paper\Review\自监督\images\image-20230706104955282.png)

对比方法背后的基本思想是将每个实例（例如，用户/项目/序列）视为一个类，然后将同一实例的变体在嵌入空间中拉得更近，并将不同实例的变体推开，其中变体是通过对原始数据进行不同的变换而创建。一般来说，同一实例的两个变体被认为是正样本对，不同实例的变体被认为是彼此的负样本。变体应该引入非必要的变化，而不是显着修改原始实例。通过最大化正对之间的一致性，同时最小化负对之间的一致性，该方法可以获得用于推荐的判别性表示。我们将对比任务表述为
$$
f_{\theta}^{\ast}=\mathop{\arg\min}\limits_{{{f_{\theta}\,,g_{\phi_{s}}}}}\,{\cal L}_{ssl}\left(g_{\phi_{s}}(f_{\theta}(\tilde{\mathcal D_{1}},\tilde{\mathcal D_{2}}))\right)
$$

### 2.3.2 Generative Methods

![image-20230708085150868](E:\Paper\Review\自监督\images\image-20230708085150868.png)

生成方法从 BERT 等掩码语言模型 (MLM) 中汲取灵感。这些模型采用自我监督任务，其中原始用户/项目配置文件是根据其损坏的版本重建的。该模型经过训练，可以从其余数据中预测一部分可用数据，其中结构和特征重建是最常见的任务。自监督任务通常表述为
$$
f_{\theta}^{\ast}=\mathop{\arg\min}\limits_{{{f_{\theta}\,,g_{\phi_{s}}}}}\,{\cal L}_{ssl}\left(g_{\phi_{s}}(f_{\theta}(\tilde{\mathcal D})),\mathcal D\right)
$$


### 2.3.3 Predictive Methods

![image-20230708085407021](E:\Paper\Review\自监督\images\image-20230708085407021.png)

SSR 中的预测方法可能看起来与生成方法类似，因为两者都涉及预测，但潜在目标是不同的。生成方法专注于预测原始数据的缺失部分，这可以被视为自我预测的一种形式。相反，预测方法从原始数据生成新样本或标签来指导借口任务。我们将现有的预测SSR方法分为两类：基于样本的和基于伪标签的。基于样本的方法旨在根据当前编码器参数预测信息样本。然后将这些预测样本反馈到编码器中以生成具有更高置信度的新样本。这种方法将自我训练（半监督学习的一种）和 SSL 结合起来。基于伪标签的方法，另一方面，通过生成器生成标签，生成器可以是另一个编码器或基于规则的选择器。然后将这些生成的标签用作地面事实来指导编码器 fθ。基于伪标签的方法可以表述如下
$$
f_{\theta}^{\ast}=\mathop{\arg\min}\limits_{{{f_{\theta}\,,g_{\phi_{s}}}}}\,{\cal L}_{ssl}\left(g_{\phi_{s}}(f_{\theta}(\mathcal D)),\tilde{\mathcal D}\right)
$$

### 2.3.4 Hybrid Methods

![image-20230708085734477](E:\Paper\Review\自监督\images\image-20230708085734477.png)

上述每种类型的方法都有其独特的优点，并且可以利用不同的自我监督信号。获得全面自我监督的一个可行策略是将各种自我监督任务结合起来，并将它们集成到单个推荐模型中。混合方法通常需要多个编码器和投影头，并且不同的自监督任务可以并行操作或协作以增强自监督信号。各种前置任务的组合通常被表述为上述类别中不同自监督损失的加权和。



## 2.4 Typical Training Schemes

尽管SSR有统一的公式，但推荐任务与前置任务在各种场景下以不同的方式耦合。在本节中，我们提出 SSR 的三种典型训练方案：联合学习（JL）、预训练和微调（PF）以及集成学习（IL）



### 2.4.1 Joint Learning (JL)

![image-20230708090733908](E:\Paper\Review\自监督\images\image-20230708090733908.png)
$$
\Theta^{*}=\operatorname*{\arg\operatorname*{min}}\limits_{f_{\theta},g_{\phi}}{\mathcal{L}}_{r e c}{\big(}g_{\phi_{r}}{\big(}f_{\theta}({\mathcal{D}}){\big)}{\big)}+\alpha{\mathcal{L}}_{s s l}{\big(}g_{\phi_{s}}{\big(}f_{\theta}({\tilde{\mathcal D}}){\big)}{\big)}
$$

### 2.4.2 Pre-training and Fine-tuning (PF)

![image-20230708091715143](E:\Paper\Review\自监督\images\image-20230708091715143.png)
$$
\begin{array}{c}{{f_{\theta_{i n i t}}=\operatorname*{\mathrm{arg\min}}\limits_{f_{\theta},g_{\phi_s}}\,{\mathcal L}_{s s l}\bigl(g_{\phi_{s}}\bigl(f_{\theta}(\tilde {D}),D\bigr)\bigr)}}\\ {{\Theta^{\ast}=\mathrm{arg\min}\,{\mathcal L}_{r e c}\bigl(g_{\phi_{r}}\bigl(f_{\theta}(D)\bigr)\bigr)}}\\ \end{array}
$$

### 2.4.3 Integrated Learning (IL)

![image-20230708092234048](E:\Paper\Review\自监督\images\image-20230708092234048.png)
$$
\Theta^{*}=\operatorname*{\arg\operatorname*{min}}\limits_{f_{\theta},g_{\phi}}{\mathcal{L}}{\big(}g_{\phi_{r}}{(}f_{\theta}({\mathcal{D}}){)},g_{\phi_{s}}{(}f_{\theta}({\tilde{\mathcal D}}){)}{\big)}
$$


# 3 DATA AUGMENTATION

先前的研究强调了数据增强在促进学习高质量、可概括的表示方面所发挥的关键作用。在深入研究 SSR 方法之前，我们概述了 SSR 中常用的数据增强技术，并将其分为三类：基于序列、基于图和基于特征。这些增强方法大多数都是任务无关且与模型无关的，并且已在各种 SSR 范式中使用



## 3.1 Sequence-Based Augmentation

## 3.2 Graph-Based Augmentation

给定用户-项目图 $
{\cal G}\:=\:\left(\cal V,\cal E\right) 
$ 和邻接矩阵 $\mathbf A$（或其他图，如用户-用户图），可以应用以下增强方法

![image-20230708094828976](E:\Paper\Review\自监督\images\image-20230708094828976.png)

### Edge/Node Dropout:

以概率 $ρ$，每条边都可以从图中删除。背后的想法是，只有部分连接对节点表示有贡献，丢弃冗余连接可以赋予表示更强的鲁棒性，这类似于裁剪方法。该方法表述为
$$
\tilde{\cal G},\tilde{\mathbf A}={\cal T}_{\mathrm{E-dropout}}({\cal G})=({\cal V},\mathrm{\mathbf m}\odot{\cal E})
$$
类似地，每个节点及其关联的边也可以从图中删除[66]、[31]、[63]。这种增强方法有望从不同的增强视图中识别有影响力的节点，其公式为
$$
\tilde{\cal G},\tilde{\mathbf A}={\cal T}_{\mathrm{N-dropout}}({\cal G})=({\cal V}\odot{\mathbf m},\mathrm{\mathcal E}\odot{\mathbf m}^{\prime})
$$

### Graph Diffusion

与基于 dropout 的方法相反，基于扩散的增强将边添加到图中以创建视图。有的研究认为缺失的用户行为包括未知的积极偏好，可以用加权的用户项目边缘来表示。因此，他们通过计算用户和项目表示的相似度来发现可能的边缘，并保留具有前 K 个相似度的边缘。该方法表述为
$$
\tilde{\cal G},\tilde{\mathbf A}={\cal T}_{\mathrm{diffusion}}({\cal G})=({\cal V},\mathrm{\mathcal E}+\tilde{\mathcal E})
$$

### Subgraph Sampling

该方法对节点和边的一部分进行采样以形成子图。许多方法可用于诱导子图，例如元路径引导的随机游走和自我网络采样。子图采样的基本思想类似于边缘丢失的思想，而子图采样通常在局部结构上进行操作。给定采样节点集 $\cal Z$，该方法可以表述为
$$
\tilde{{\cal G}},\tilde{\mathbf{A}}={\cal T}_{\mathrm{sampling}}({\cal G})=({{\cal Z}}\in{\cal V},{\mathrm A}[{\cal Z},{\cal Z}])
$$


## 3.3 Feature-Based Augmentation

### Feature Dropout

[26], [65], [72], [73], [74], [75] 特征丢失与边缘丢失类似，它随机丢弃一小部分特征，公式为：
$$
\tilde{\mathbf{X}}={\cal T}_{\mathrm{F-dropout}}({\mathbf{X}})=\mathrm{\mathbf X}\odot{\mathbf M}
$$

### Feature Shuffling 

[30], [29], [76] 切换特征矩阵X中的行和列。通过随机改变上下文信息，X被破坏以产生增强。该方法可以表述为：
$$
\tilde{\mathbf{X}}={\cal T}_{\mathrm{shuffling}}({\mathbf{X}})=\mathrm{P}_r\mathrm{X}\mathrm{P}_c
$$

### Feature Clustering

[77]、[78]、[79] 提出的将 CL 与聚类相结合，假设特征/表示空间中存在原型，并且每个用户/项目表示在语义上应该与分配的原型相似，其中原型是通过无监督学习的像EM算法一样学习。其公式为
$$
\tilde{\bf C}={\cal T}_{\mathrm{clustering}}({\bf X})=\operatorname{EM}({\bf X},\cal C)
$$

### Feature Mixing

[80]、[65] 将原始用户/项目特征与其他用户/项目或先前版本的特征混合，以合成信息丰富的负面/正面示例[81]。它通常按以下方式对两个样本进行插值：
$$
\tilde{\bf x}_{i}={\mathcal T}_{\mathrm{mixing}}({\bf x}_{i})=\alpha{\bf x}_{i}+({1-\alpha}){\bf x}_{j}^{\prime}
$$

### Feature Perturbation 

[82]、[83]将随机噪声添加到原始用户/项目表示中。由于添加噪声的幅度非常小，增强表示保留了原始表示的大部分信息，同时引入了一些差异。该方法表述为：
$$
\tilde{\bf x}_{i}\,=\,{\cal T}_{\mathrm{Perturbation}}\!\left({\bf x}_{i}\right)\,=\,{\bf x}_{i}\,+\,\lambda\Delta_{i}
$$


# 4 CONTRASTIVE METHODS

各种数据增强方法和数据类型催生了各种形式的对比性前置任务。根据自监督信号的来源，这些任务可以分为三组:**结构级对比度**、**特征级对比度**和**模型级对比度**。表1总结了所调查的对比方法



## 4.1 Structure-Level Contrast

用户行为数据通常表示为图或序列，其中对图/序列结构的轻微扰动可能会导致类似的语义。通过对比不同的结构，可以获得结构扰动的共同不变性作为自监督信号。我们遵循[35]、[36]提出的分类法，将结构层次对比分为两类：同尺度对比和跨尺度对比。同尺度对比涉及相同尺度的两个物体的视图，并进一步分为两个层次：局部-局部和全局-全局。跨尺度对比涉及不同尺度的两个对象的视图，并进一步分为局部-全局和局部上下文。在图结构中，局部指节点，全局指图，而在序列结构中，局部指项目，全局指序列



### 4.1.1 Local-Local Contrast

这种类型的对比伴随着基于图的 SSR 模型，以最大化用户/项目节点表示之间的互信息，其公式为
$$
f_{\theta}^{*}=\operatorname*{\arg\mathrm{min}}\limits_{f_{\theta},g_{\phi_s}}\,{\mathcal L}_{\mathcal{MI}}(g_{\phi_{s}}(\tilde{\bf h}_{i},\tilde{\bf h}_{j}))
$$


对于局部对比度，**基于丢失的增强**是创建扰动局部视图的最优选方法:

SGL [31] 作为代表性模型，将节点丢失、边缘丢失和随机游走增强应用于用户-项目二分图。它使用相同类型的增强算子生成两个增强图，并使用共享图 LightGCN 编码器 fθ [84] 学习节点嵌入。节点级对比度使用批内负采样的InfoNCE损失[15]进行优化，并与贝叶斯个性化排名（BPR）损失[85]联合优化以进行推荐

DCL[86] 采用随机边缘丢失来扰乱节点的 L 跳自我网络，从而产生两个增强的邻域子图。然后，它最大化在两个子图上学习的节点表示之间的一致性

HHGR [62] 提出了一种用于组推荐的双尺度节点丢失方法[87]。该方法包括粗粒度和细粒度的丢弃方案，分别从所有组中删除一部分用户节点并仅从特定组中丢弃随机选择的成员节点。然后，它最大化从具有不同丢弃粒度的这两个视图中学习到的用户节点表示之间的互信息

KGCL[88] 将 dropout 应用于知识图，并提出了一种知识感知对比方法，该方法对比从用户项图和知识图的增强中学习到的节点表示



子图采样是另一种创建局部图对比度的流行方法:

CCDR [70]将对比学习应用于跨域推荐，使用两种类型的对比任务：CL 内和 CL 间。 CL 内任务类似于 DCL [86] 中的对比任务，使用图注意网络 [89] 作为编码器在目标域中进行。 CL 间任务旨在最大化源域和目标域中学习的同一对象的表示之间的互信息。并发工作 PCRec [71] 还将跨域推荐与 CL 连接起来，使用随机游走对 r-hop 自我网络进行采样以增强数据。它通过对比采样子图在源域中预训练 GIN [90] 编码器，然后使用交互数据微调矩阵分解 (MF) [91] 模型以在目标域中进行推荐。



### 4.1.2 Global-Global Contrast

全局级对比度通常用于顺序推荐模型



### 4.1.3 Local-Global Contrast

局部-全局对比旨在将高级全局信息编码为局部结构表示并统一全局和局部语义。它经常用于图学习场景，可以表示为
$$
f_{\theta}^{*}=\operatorname*{\arg min}_{f_{\theta},g_{\phi_s}}{\cal L}_{\cal MI}(g_{\phi_{s}}(\tilde{\bf h},{\cal R}(f_{\theta}(\tilde{\cal G},\tilde{\bf A})))
$$
EGLN [67] 提出通过将合并的用户-项目对表示与全局表示进行对比来实现局部-全局一致性，全局表示是所有用户项目对表示的平均值。它还采用图扩散进行数据增广，通过计算用户和项目之间的相似度获得增广图邻接矩阵，保留top-K相似度。矩阵和用户/项目表示迭代地相互学习并通过图形编码器进行更新

BiGI [69] 执行局部-全局对比，但在生成用户-项目对表示时，仅对其 $h-$hop 子图进行采样以进行特征聚合

在 HGCL [93] 构建了用户和项目节点类型特定的同构图。对于每个齐次图，它使用 DGI [17] 管道最大化图的局部补丁和整个图的全局表示之间的互信息。它还提出了跨类型对比来测量不同类型同质图的局部和全局信息



### 4.1.4 Local-Context Contrast

在基于图和基于序列的场景中观察到局部上下文对比，其中上下文是通过采样自我网络或聚类来构建的。这种类型的对比度可以表述为：
$$
f_{\theta}^{*}=\operatorname*{\arg min}_{f_{\theta},g_{\phi_s}}{\cal L}_{\cal MI}(g_{\phi_{s}}({\bf h}_{i},{\cal R}(f_{\theta}(\mathcal C_j)))
$$
NCL[78] 设计了一个原型对比目标来捕获用户/项目与其原型之间的相关性，该原型代表一组语义邻居。通过使用 K-means 算法对所有用户或项目嵌入进行聚类来获得原型，并使用 EM 算法递归地调整原型。

ICL [77] 有一个类似的管道，但它是为顺序推荐而设计的，其中语义原型被建模为用户意图，所属序列是原型的本地视图

MHCN [30] 通过定义三种类型的三角社交关系并使用多通道超图编码器对其进行建模，将 SSL 应用到社交推荐 [94] 中。对于每个通道中的每个用户，MHCN 分层最大化用户表示、用户自我超图表示和全局超图表示之间的互信息

HCCF [95] 提出参数化超图依赖矩阵，而不是手动定义超图结构。然后，它对比从用户-项目图和参数化超图导出的表示

SMIN [96] 使用具有不同顺序的上下文聚合的用户项邻接矩阵链将节点与其上下文进行对比

$\bf S^3$-Rec [26] 应用项目屏蔽和项目裁剪来增强序列，并设计了四个对比任务来预训练用于下一个项目预测的双向 Transformer：项目属性互信息最大化（MIM）、序列项目 MIM、序列-属性MIM和序列-序列MIM



## 4.2 Feature-Level Contrast



## 4.3 Model-Level Contrast

前两类从数据角度提取自监督信号，但它们并不是以端到端的方式实现。另一种方法是保持输入不变并动态修改模型架构以动态增强视图对。这些模型级增强之间的对比被表述为：



$$
f_{\theta}^{*}=\operatorname*{\arg min}_{f_{\theta},g_{\phi_s}}{\cal L}_{\cal MI}\big(g_{\phi_{s}}(f_{\theta^{\prime}}({\cal{D}}),f_{\theta^{\prime\prime}}({\cal D}))\big)
$$
神经元掩蔽是用于扰动模型的常用技术:

DuoRec [73] 将不同的 dropout mask 应用到基于 Transformer 的主干上，以实现两个模型级表示增强，从而最大化两个表示之间的互信息。该方法看似简单，但在下一项预测任务中显示出显着的性能

SimGCL [82] 和 XSimGCL [83] 将随机均匀噪声添加到隐藏表示中以进行增强，从而产生更均匀的节点表示，从而减轻流行度偏差问题 [39]。调整噪声大小可以对表示均匀性进行更细粒度的调节，从而在推荐准确性和模型训练效率方面优于 SGL



## 4.4 Contrastive Loss

对比损失是机器学习界的一个研究热点，并且在SSR中也受到越来越多的关注。一般来说，对比损失的优化目标是最大化两个表示 $\mathbf h_{i}$ 和 $\mathbf h_{j}$ 之间的互信息（MI），定义为
$$
{\cal MI}\left({\bf h}_{i},{\bf h}_{j}\right)={\mathbb{E}}_{P}({\bf h}_{i},{\bf h}_{j})\log\frac{P\left({\bf h}_{i},{\bf h}_{j}\right)}{P\left({\bf h}_{i}\right)P\left({\bf h}_{j}\right)}
$$
近年来，对比性方法因其在数据扩充和前置任务集合方面的灵活性得到了迅速发展，并覆盖了大部分推荐主题。虽然对比SSR在用轻量级架构改进推荐方面表现出了显著的有效性，但它经常受到高质量数据增强的未知标准[57]的影响。现有的对比方法大多基于任意数据增强，通过试错来选择。既没有对它们如何和为什么工作的严格理解，也没有规则或指导方针清楚地说明什么是好的增强。此外，一些被认为有用的常见增强方法，最近甚至被证明对推荐性能有负面影响[82]。因此，在不知道增强是什么信息的情况下，对比任务可能会失败



# 5 GENERATIVE METHODS

# 6 PREDICTIVE METHODS

# 7 HYBRID METHODS

# 8 SELFREC: A LIBRARY FOR SELF-SUPERVISED RECOMMENDATION

# 9 EXPERIMENTAL FINDINGS

## 9.1 Comparison of Data Augmentations in CL

![image-20230709155952277](E:\Paper\Review\自监督\images\image-20230709155952277.png)

- 这两种情况都表明，特征级别的增强非常有效。具体来说，向特征添加噪声平均会带来最高的改进
- 结构级别的增强对于稀疏数据集效果较差，甚至可能导致顺序场景中的性能下降。然而，它们可以对更密集的数据集中的性能做出积极贡献
- 模型级别增强的有效性因不同数据集而异。一些数据集显示出相当大的改进，而另一些数据集则显示出很小的改进
- 顺序对比 SSR 的增强不如其对应的图形有效。造成这种情况的一个可能原因是项目转换之间缺乏清晰的语义，这可能会限制 Transformer 结构的潜力



## 9.2 Comparison of SSR Models

不同的自监督任务从不同的角度改进推荐模型。为了确定 SSR 中最有效的范例，我们在图和顺序场景中比较了几种流行的 SSR 模型。我们对表 8 和表 9 中的结果进行分析得出以下结论：（请注意，我们使用 (G, C, P) 来表示比较方法的类别，分别指生成法、对比法和预测法。我们还手动构建了一个基于 LightGCN 的生成图模型（AdjRecons），采用结构生成任务进行公平比较，因为不存在这样的已发布模型。）

- 在图场景中，对比SSR方法表现出优越的推荐性能。 SGL 和 SimGCL 都显着改进了 LightGCN，但 SimGCL 由于其更有效的基于噪声的特征增强而显示出更大的优越性。相比之下，这种情况下的预测 SSR 方法令人失望，甚至大大降低了性能。我们认为这是因为预测 SSR 需要更多信息（例如属性）来创建更强且有利的自我监督信号，但所使用的数据集不提供属性信息。生成 SSR 方法的性能介于对比 SSR 和预测 SSR 之间，但仍取得了不错的改进
- 在顺序场景中，对比和预测自监督推荐方法显示出类似的性能改进，而生成方法 BERT4Rec 获得了令人失望的结果，甚至远低于 SASRec 的结果。我们认为这是因为 BERT4Rec 中没有进行微调。它仅通过双向屏蔽项预测进行预训练。然而，下一项预测和双向屏蔽项预测之间可能存在差距
- 与图场景中的SSR方法相比，顺序SSR方法仍有改进的空间。我们的研究结果表明，许多顺序 SSR 模型并不像原始论文中报道的那么有效




# 10 DISCUSSION

## 10.1 Theory for Augmentation Selection

虽然数据增强对于提高 SSR 性能至关重要，但当前大多数方法都依赖于从 CV、NLP 和图学习等其他领域借用的启发式方法。然而，这些方法无法无缝移植到推荐中来处理与场景紧密耦合且混有噪声和随机性的用户行为数据。此外，大多数方法基于启发式增强数据，并通过繁琐的试错来搜索合适的增强。尽管有一些理论试图揭开对比学习中视觉视图选择的神秘面纱[138]、[57]，但推荐中增强选择的原理却很少被研究。因此，迫切需要一个坚实的推荐特定理论基础，以简化选择过程并将人们从繁琐的试错工作中解放出来

## 10.2 Explainable Self-Supervised Recommendation

尽管现有 SSR 模型取得了有希望的结果，但在大多数情况下，其性能提升背后的机制在理论上并不合理。这些模型通常被认为是黑匣子，其主要目标是实现更高的性能。然而，增强和自我监督目标等组件缺乏可靠的可解释性来证明其有效性。 [82]中最近的实验表明，一些先前被认为具有信息性的图形增强甚至可能会损害性能。此外，尚不清楚这些模型是否会牺牲其他属性（例如稳健性）来提高性能。为了创建更可靠的 SSR 模型，了解他们学到了什么以及模型如何通过自我监督训练进行更改至关重要

## 10.3 Attacking and Defending Self-Supervised Recommendation Models

由于开放性，推荐系统很容易受到数据中毒攻击，这种攻击会将故意制作的用户-项目交互数据注入到模型训练集中，以篡改模型参数并操纵推荐结果[139]、[140]。监督推荐系统的攻击和相应的防御方法已经得到了充分的研究。然而，SSR 模型是否对此类攻击具有鲁棒性仍然未知。我们还注意到，一些开创性的工作尝试攻击视觉和图形分类任务中的预训练编码器[141]、[142]。为了保证SSR模型的鲁棒性，开发新的攻击策略和相应的防御机制是紧迫而重要的

## 10.4 On-Device Self-Supervised Recommendation

现代推荐系统通过完全基于服务器的操作来满足数百万用户的需求，这会产生巨大的碳足迹并引发隐私问题。去中心化推荐系统[143]、[144]已经作为一种解决方案出现，部署在智能手机等资源受限的设备上。然而，设备上的推荐系统受到高度压缩的模型大小和有限的标记数据的阻碍。 SSL 可以潜在地解决这些问题，特别是与知识蒸馏技术 [145]、[133]、[146] 相结合以补偿准确性下降时。目前，设备上 SSR 的探索还较少，值得进一步研究

## 10.5 Towards General-Purpose Pre-Training

在工业领域，推荐系统处理多模态数据和多样化场景。深度推荐模型针对各种推荐任务跨不同模式（例如视频、图像和文本）进行训练[147]。然而，训练和任务往往是独立的，并且需要大量的计算资源。考虑到跨模态数据的相关性，探索通过大规模数据的多模态自监督学习预先训练的通用推荐模型是很自然的。这些模型可以通过廉价的微调来适应多个下游推荐任务，这使得它们对于训练数据稀疏的场景特别有用。尽管人们一直在努力开发通用推荐模型[115]、[148]、[54]、[149]，但它们大多以类似 BERT 的方式进行训练，具有相似的架构。值得研究更高效的训练策略和有效的模型架构

