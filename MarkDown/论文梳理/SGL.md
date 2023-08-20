**发现问题**

1. high-degree 节点对表示学习产生很大的影响，从而恶化了长尾项目的推荐
2. representation 很容易受到噪声交互的影响

**基于 GCN 的推荐系统的局限性**

- 监督信号的稀疏，在推荐系统中监督信号指用户-项目之间的交互
- 数据的倾斜分布，数据的交互通常呈现幂律分布，也称之为长尾分布，因此基于 GCN 的推荐系统很容易受出现频次较多的项目的影响
- 交互中的噪音，由于反馈大多呈隐式反馈，所以交互中包含不明确的噪音，影响 GCN 的表示学习

**SSL 在推荐系统中的应用手段**

1. 数据增强，为每个节点生成多个视图
2. 对比学习，最大化同一节点不同视图之间的一致性，为此构建三种不同的邻接矩阵：node dropout，edge dropout，random walk

**简单总结贡献**

- 设计了一种新的学习范式——SGL
- 减轻程度偏差和提高对交互噪声的鲁棒性之外，从理论上证明，SGL 本质上鼓励从 hard 负例中学习，并由 softmax 损失函数中的温度超参数控制
- 实验

**回顾了基于 GCN 的协同过滤模型**
$$
{\mathbf Z}^{\left(l\right)}={H}({\mathbf Z}^{\left(l-1\right)}, \mathcal G)
$$
${\mathbf Z}^{\left(l\right)}$ 表示地 $l$ 层的节点表示
$$
\mathbf{z}_{u}^{(l)}=f_{\mathrm{combine}}(\mathbf{z}_{u}^{(l-1)},f_{\mathrm{aggregate}}(\{\mathbf{z}_{i}^{(l-1)}\vert i\in {\cal N}_{u}\}))
$$
其中 $f_{\mathrm{combine}}$ 和 $f_{\mathrm{aggregate}}$ 有多种设计
$$
{\bf z}_{u}=f_{\mathrm{readout}}\Bigl(\{{\bf z}_{u}^{(l)}|l=\left[0,\cdot\cdot\cdot,L\right]\}\Bigr)
$$
$f_{\mathrm{readout}}$ 的设计一般有保留最后一层、串联、加权和
$$
\hat{y}_{u i}={\bf z}_{u}^{\top}{\bf z}_{i}
$$
用最终表示来建立预测结果，一般采用的方案是内积
$$
{\mathcal{L}}_{m a i n}=\sum_{(u,i,j)\in {\cal O}}-\log\sigma({\hat{y}}_{u i}-{\hat{y}}_{u j})
$$
Bayesian Personalized Ranking —— 贝叶斯个性化排名
$$
{\cal O}=\{(u,i,j)|(u,i)\in{\cal O}^{+},(u,j)\in{\cal O}^{-}\}
$$
**整体架构**

![image-20230807101701067](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230807101701067.png)

**为什么不能直接借鉴 CV 和 NLP 的数据增强**

1. user 和 item 的特征是离散的，所以图像上的随即裁剪、旋转、模糊都不适用
2. 交互图中的 user 和 item 本质上是相互连接和依赖的

**三种数据增强的具体做法**

Node Dropout：以概率 𝜌 从图中丢弃每个节点及其连接的边
$$
s_{1}(\mathcal{G})=({\bf M}^{\prime}\odot {\cal V},\mathcal{E}),\quad s_{2}(\mathcal{G})=(M^{\prime\prime}\odot {\cal V},\mathcal{E})
$$
${\bf M}^{\prime},{\bf M}^{\prime\prime}\;\in\;\{0,1\}^{|{\cal E}|} $ 是两个掩码向量，应用在节点集 $\cal V$ 上来生成两个子图，即多视角识别出“重要”的节点

Edge Dropout：以概率 𝜌 从图中丢弃边
$$
s_{1}(\mathcal{G})=({\cal V},{\bf M}_1\odot\mathcal{E}),\quad s_{2}(\mathcal{G})=({\cal V},{\bf M}_2\odot\mathcal{E})
$$
${\bf M}_{1},{\bf M}_{2}\;\in\;\{0,1\}^{|{\cal E}|} $ 同样是是两个掩码向量，应用在边集 $\cal E$ 上

Random Walk：上述两种方法生成的子图是所有卷积层之间所共享的，随机游走为不同的层之间生成不同的子图，也可以认为为每个节点构建单独的子图
$$
s_{1}(\mathcal{G})=({\cal V},{\bf M}_1^{(l)}\odot\mathcal{E}),\quad s_{2}(\mathcal{G})=({\cal V},{\bf M}_2^{(l)}\odot\mathcal{E})
$$
**对比学习范式**

在建立节点的增强视图之后，将同一节点的视图(views)视为正对，同时将与之不同的节点的视图视为负对。这么做的动机是为了促进同一节点不同视图之间预测的一致性，放大不同节点之间的差异
$$
{\cal L}_{s s l}^{u s e r}=\sum_{u\in{\cal U}}-\log\frac{\exp(s({\bf z}_{u}^{\prime},{\bf z}_{u}^{\prime\prime})/\tau)}{\sum_{v\in{\cal U}}\exp(s({\bf z}_{u}^{\prime},{\bf z}_{v}^{\prime\prime})/\tau)},
$$
$s(·)$ 是衡量两个向量之间的相似度，一般设置为余弦相似度函数，$\tau$ 是温度超参数。同理可获得 item 的 loss，所以总 loss 为：${\cal L}_{s s l}={\cal L}_{s s l}^{u s e r} + {\cal L}_{s s l}^{item}$

**联合(多任务)训练**
$$
{\mathcal{L}}={\mathcal{L}}_{m a i n}+\lambda_{1}{\mathcal{L}}_{s s l}+\lambda_{2}\left\|\Theta\right\|_{2}^{2}
$$
**SGL 背后的理论支撑**

它具有内在的能力来执行 hard 负样本挖掘，这为优化过程提供了大量有意义的梯度，并指导节点表示学习。hard 负节点，其表示与正节点 𝑢 相似（即 0 < 𝑥 ≤ 1），因此很难在潜在空间中区分 𝑣 和 𝑢

**SGL 的优越性**

1. 长尾推荐
2. 训练效率
3. 对噪声的鲁棒性