# INTRODUCTION

当前大多数图对比学习（GCL）方法都采用基于启发式的对比视图生成器来最大化输入正对之间的互信息并推开负实例。为了构建扰动视图，SGL 被提出通过使用随机增强策略（例如 node dropout 和 edge perturbation）破坏 user-item 交互图的结构信息来生成正视图的节点对，SimGCL 提出带有随机噪声扰动的嵌入增强，为了识别节点（用户和项目）的语义邻居，引入了 HCCF 和 NCL ，以追求结构相邻节点和语义邻居之间的一致表示

尽管它们很有效，但最先进的对比推荐系统仍存在一些固有的局限性

i）随机扰动的图增强可能会丢失有用的结构信息，从而误导表示学习

ii）启发式引导表示对比方案的成功很大程度上建立在视图生成器的基础上，这限制了模型的通用性并且容易受到嘈杂的用户行为的影响

iii）当前大多数基于 GNN 的对比推荐器都受到过度平滑问题的限制，导致无法区分表示

# METHODOLOGY

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913101848885.png" alt="image-20230913101848885" style="zoom:80%;" />



**LOCAL GRAPH DEPENDENCY MODELING**

协同过滤的常用方法是为每一个 user $u_i$ 和 item $v_j$ 分配一个 embedding vector ${\bf e}_{i}^{\left(u\right)},{\bf e}_{j}^{\left(v\right)}\in\;{\mathbb R}^{d} $，其中 $d$ 是嵌入大小，所有 user 和 item 嵌入的集合定义为 ${\bf E}^{\left(u\right)}\in\;{\mathbb R}^{I\times d},{\bf E}^{\left(v\right)}\in\;{\mathbb R}^{J\times d} $，$I$ 和 $J$ 分别是用户数和物品数，之后采用 GCN 来聚合每个节点的邻居信息
$$
z_{i,l}^{(u)}=\sigma(p(\tilde{\cal A}_{i,:)}\cdot{\bf E}_{l-1}^{(v)}),\quad z_{j,l}^{(v)}=\sigma(p(\tilde{\cal A}_{:,j})\cdot{\bf E}_{l-1}^{(u)})
$$
$\tilde{\mathcal A}$ 是归一化邻接矩阵，在其上执行边缘 dropout，表示为 $p$(·)，以减轻过度拟合问题，节点的 final embedding 是其所有层嵌入的总和，用户 $u_i$ 和项目 $v_j$ 的最终嵌入之间的内积预测 $u_i$ 对 $v_j$ 的偏好：
$$
e_{i}^{(u)}=\sum_{l=0}^{L}z_{i,l}^{(u)},\quad e_{j}^{(v)}=\sum_{l=0}^{L}z_{j,l}^{(v)},\quad\hat{y}_{i,j}=e_{i}^{(u)\top}e_{j}^{(v)}
$$


**EFFICIENT GLOBAL COLLABORATIVE RELATION LEARNING**

采用 SVD 方案为通过全局结构学习增强图对比学习的推荐能力，以从全局角度有效地提取重要的协作信号

首先对 normalized adjacency matrix $\tilde {\mathcal A}$ 进行 SVD ，$\tilde {\mathcal A}=USV^{\top}$,通过 SVD 的方式分解原矩阵，$S$ 是一个 $I\times J$ 的对角矩阵，其值是 $\tilde {\mathcal A}$ 的奇异值(singular values)。最大奇异值通常与矩阵的主成分相关，也就是说对矩阵的贡献越大。所以本文保留前 $q$ 个奇异值，并利用保留的数值重构 $\tilde {\mathcal A}$，得到 $\hat {\mathcal A}=U_qS_qV_q^{\top}$

基于 SVD 的图结构学习的优点有两个

i) 它通过识别对用户偏好表示重要且可靠的用户-项目交互来强调图的主要组成部分

ii) 生成的新图结构通过考虑每个 user-item pair 来保留全局协作信号

然后基于重构的 $\tilde {\mathcal A}$ 再执行消息的传播
$$
g_{i,l}^{(u)}=\sigma(\hat{\cal A}_{i,:}\cdot{\bf E}_{l-1}^{(v)}),\quad g_{j,l}^{(v)}=\sigma(\hat{\cal A}_{:,j}\cdot{\bf E}_{l-1}^{(u)})
$$
然而，在大型矩阵上执行精确的 SVD 成本非常高，这使得处理大规模 user-item 矩阵变得不切实际。因此，采用随机SVD
$$
\hat{U}_{q},\hat{S}_{q},\hat{V}_{q}^{\top}=\mathrm{ApproxSVD(\hat{{\mathcal A}},q),\quad\hat{{\mathcal A}}_{SVD}}=\hat{U}_{q}\hat{S}_{q}\hat{V}_{q}^{\top}
$$
所以重写传播公式
$$
G_{l}^{(u)}=\sigma(\hat{\cal A}_{SVD}{\bf E}_{l-1}^{(v)})=\sigma(\hat{U}_{q}\hat{S}_{q}\hat{V}_{q}^{\top}{\bf E}_{l-1}^{(v)}),\quad G_{l}^{(v)}=\sigma(\hat{\cal A}_{SVD}{\bf E}_{l-1}^{(u)})=\sigma(\hat{U}_{q}\hat{S}_{q}\hat{V}_{q}^{\top}{\bf E}_{l-1}^{(u)})
$$


**SIMPLIFIED LOCAL-GLOBAL CONTRASTIVE LEARNING**

传统的 GCL 方法（例如 SGL 和 SimGCL）通过构造两个额外的视图来对比节点嵌入，而从原始图（主视图）生成的嵌入不直接参与 InfoNCE 损失，所以本文直接在 main-view 和 SVD-augmented view 上进行 embeddings CL
$$
{\mathcal{L}}_{s}^{(u)}=\sum_{i=0}^{I}\sum_{l=0}^{L}-\log{\frac{\exp(s(z_{i,l}^{(u)},g_{i,l}^{(u)}/\tau))}{\sum_{i^{\prime}=0}^{I}\exp(s(z_{i,l}^{(u)},g_{i^{\prime},l}^{(u)})/\tau)}}
$$
为了防止过拟合，在每个批次中实施随机节点丢弃，以排除某些节点参与对比学习，然后定义 Loss Function
$$
{\mathcal{L}}={\mathcal{L}}_{r}+\lambda_{1}\cdot({\mathcal{L}}_{s}^{(u)}+{\mathcal{L}}_{s}^{(v)})+\lambda_{2}\cdot\|\Theta\|_{2}^{2};\ \ \ \ {\mathcal{L}}_{r}=\sum_{i=0}^{I}\sum_{s=1}^{S}\operatorname*{max}(0,1-{\hat{y}}_{i,p_{s}}+{\hat{y}}_{i,n_{s}})
$$

# EVALUATION

**How does LightGCL perform on different datasets compared to various SOTA baselines?**+

将所有基线的超参数调整在原始论文建议的范围内，除了所有模型的以下固定设置：嵌入大小设置为32；批量大小为256

归因于 CL 学习均匀分布嵌入的有效性；归因于通过注入全局协作上下文信号来有效增强图对比学习

**How does the lightweight graph contrastive learning improve the model efficiency?**

**How does our model perform against data sparsity, popularity bias and over-smoothing?**

t-SNE

**How does the local-global contrastive learning contribute to the performance of our model?**

**How do different parameter settings affect our model performance?**

