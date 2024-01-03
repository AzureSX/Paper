设计一个合适的对比学习框架仍然很重要，因为需要仔细考虑对比学习和知识感知推荐的特性以求平衡，需要解决以下基本问题

1. 如何设计合适的对比机制？由于异构性，设计的模型自然需要同时处理多种类型的节点（user/item/entity）和关系（user-item/item-entity）
2. 如何构建正确的对比学习视图？一种直接的方法是将输入的 user-item-entity graph 增强（或破坏）为 graph view，并将其与原始图进行对比。然而，仅仅考虑 KGR 的全局视图（user-item-entity graph）是远远不够的，因为它无法充分利用丰富的协作信息（item-user-item co-occurrence）和语义信息（item-user-item co-occurrence）。显然，仅在粗粒度级别使用一个 graph view 使得难以充分利用丰富的协作和语义信息进行推荐



<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230912093715891.png" alt="image-20230912093715891" style="zoom:80%;" />

# Multi Views Generation

与以前仅考虑全局用户实体图的方法不同，本文提出通过联合考虑局 local/global-level 的视图，以更全面、更细粒度的方式进行学习。首先根据 item-item 关系的不同类型，将 user-item-entity graph 分为 user-item graph 和  item-entity graph

- 对于 user-item graph，将其视为 **collaborative view**，旨在挖掘项目之间的协作关系，即 item-user-item co-occurrences
- 对于 item-entity graph，将其视为 **semantic view**，旨在探索项目之间的语义相似性，即 item-entity-item co-occurrences
- 对于 original user-item-entity graph，将其视为 **structural view**，旨在保留完整的路径信息，即 user-item-entity 的远程连通性

在构建 semantic view 的同时，为了明确考虑 item-item 间语义关系，本文构建一个具有关系感知聚合机制的 𝑘-Nearest-Neighbor item-item semantic graph 𝑆，该机制保留邻居实体和关系信息。 𝑆 中的每个条目 $S_{ij}$ 表示 item i 和 item j 之间的语义相似度。特别是，$S_{ij}=0$ 表示它们之间没有联系
$$
\mathbf{e}_{i}^{\left(k+1\right)}={\frac{1}{\left|{\mathcal N}_{i}\right|}}\sum_{\left(r,v\right)\in {\mathcal N}_{i}}\mathbf{e}_{r}\odot\mathbf{e}_{v}^{\left(k\right)}
$$

$$
\mathbf{e}_{v}^{\left(k+1\right)}={\frac{1}{|{\mathcal N}_{v}|}}\left(\sum_{\left(r,v\right)\in {\mathcal N}_{v}}\mathbf{e}_{r}\odot\mathbf{e}_{v}^{\left(k\right)}+\sum_{\left(r,i\right)\in {\mathcal N}_{v}}\mathbf{e}_{r}\odot\mathbf{e}_{i}^{\left(k\right)}\right)
$$

其中 $\mathbf{e}_{i}^{\left(k\right)}$ 和 $\mathbf{e}_{v}^{\left(k\right)}$ 分别表示 item $i$ 和 entity $v$ 的 representation，对于每个三元组 (𝑖, 𝑟, 𝑣)，关系消息 $\mathbf{e}_{r}\odot\mathbf{e}_{v}^{\left(k\right)} $ 被设计用于通过 projection or rotation operator 对关系 𝑟 进行建模来表示三元组的不同含义，简单

受之前工作的启发，基于 cosine similarity 构建 item-item similarity graph
$$
S_{i j}=\frac{\left({\bf e}_{{i}}^{\left(K^{\prime}\right)}\right)^{\top}{\bf e}_{{j}}^{\left(K^{\prime}\right)}}{\left\|{\bf e}_{{i}}^{\left(K^{\prime}\right)}\right\|\left\|{\bf e}_{{j}}^{\left(K^{\prime}\right)}\right\|}
$$
接下来，在完全连接的 item-item graph 上进行 𝑘NN 稀疏化，减少计算要求、可行的噪声和不重要的边缘，"kNN sparsification" 是指通过一种方法来减少 k-最近邻（k-nearest neighbors）算法的计算和存储开销，同时保持模型的性能
$$
\widehat{S}_{i j}=\left\{\begin{array}{l l}{{S_{i j},}}&{{S_{i j}\in\;\mathrm{top-k\,(S_{i})}\,,}}\\ {{0,}}&{{\mathrm{otherwise,}}}\end{array}\right.
$$
$\widehat{S}_{i j}$ 是 sparsified and directed graph adjacency matrix，为了缓解梯度爆炸或消失问题，邻接矩阵被归一化
$$
\widetilde{S}=(D)^{-\frac{1}{2}}\widehat{S}\left(D\right)^{-\frac{1}{2}}
$$
最终得到 item-item semantic graph $S$ 及其 normalized sparsified adjacency matrix $\widetilde S$

目前已有的 views

- user-item interaction graph $\mathbf Y$ for **collaborative view**
- item-item semantic graph $S$ for **semantic view**
- user-item-entity whole graph for **structural view**

简单总结一下，所谓的 multi views 只有 semantic view 是本文提出的，other views 是原本自带的只是换了一个说法，套了一层皮



# Local-level Contrastive

**Collaborative View Encoder**

在 collaborative view 上利用 Light-GCN 进行信息聚合
$$
\begin{array}{c}{{\displaystyle{\bf e}_{u}^{(k+1)}=\sum_{i\in{\cal N}_{u}}\frac{1}{\sqrt{|{\cal N}_{u}||{\cal N}_i|}}\mathrm{e}_{i}^{(k)}}}\\ {{\displaystyle{\bf e}_{i}^{(k+1)}=\sum_{u\in{\cal N}_{i}}\frac{1}{\sqrt{|{\cal N}_{u}||{\cal N}_i|}}\mathrm{e}_{u}^{(k)}}}\end{array}
$$
然后将不同层的 representation 求和为 local collaborative representations ${\mathbf z}^c_i$ and ${\mathbf z}^c_u$ 
$$
{\mathbf{z}}_{u}^{c}=\mathbf{e}_{u}^{(0)}+\cdot\cdot\cdot+\mathbf{e}_{u}^{(K)},\quad {\mathbf{z}}_{i}^{c}=\mathbf{e}_{i}^{(0)}+\cdot\cdot\cdot+\mathbf{e}_{i}^{(K)}
$$
**Semantic View Encoder**

在 semantic view 上依然采用 Light-GCN 进行聚合操作，并融入 item-item affinities
$$
\mathrm{e}_{i}^{(l+1)}=\sum_{j\in{\cal N}(i)}\widetilde{S}\mathrm{e}_{j}^{(l)}
$$
求和每层
$$
\mathbf{z}_{i}^{s}=\mathbf{e}_{i}^{(0)}+\mathbf{\cdot\cdot\cdot}+\mathbf{e}_{i}^{(L)}
$$
**Local-level Cross-view Contrastive Optimization**

通过 collaborative/semantic views，分别获得了两个 embeddings ${\mathbf{z}}_{i}^{c}$ 和 ${\mathbf{z}}_{i}^{s}$，为了将它们映射到计算对比损失的空间中，embeddings 首先被输入到具有一个隐藏层的 MLP 中，这步在其他工作中被定义为投影层
$$
\begin{array}{l}{{\mathbf {z}_{i}^{c}\_{\mathrm p}={W}^{(2)}\sigma\left({W}^{(1)}\mathbf {z}_{i}^{c}+b^{(1)}\right)+b^{(2)}}}\\ {{ {\mathbf {z}_{i}^{s}\_{\mathrm p}}={W}^{(2)}\sigma\left({ W}^{(1)}\mathbf {z}_{i}^{s}+b^{(1)}\right)+b^{(2)}}}\end{array}
$$
然后，**受到其他领域工作的启发**，在这里定义正样本和负样本，对于一个视图中的任何节点，另一个视图学习到的相同节点嵌入形成正样本；并且在两个视图中，除它之外的节点嵌入自然被视为负样本，这里定义了两个负样本

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230912202617023.png" alt="image-20230912202617023" style="zoom:80%;" />

# Global-level Contrastive Learning

**Structural View Encoder**

为了在 structural view 下对结构信息（即路径的多样性）进行编码，**受之前工作的启发**，这里提出了一种路径感知的GNN，它聚合 𝐿′ 次的邻近信息，同时保留路径信息，即 long-range 连接，感觉在讲故事
$$
\begin{array}{l}{{\displaystyle{\bf e}_{u}^{(l+1)}=\frac{1}{|{\cal N}_{u}|}\sum_{i\in{\cal N}_{u}}{\bf e}_{i}^{(l)},}}\\ {{\mathbf{e}_{i}^{(l+1)}=\frac{1}{|{\cal N}_{i}|}\sum\limits_{(r,v)\in{\cal N}_{i}}\beta(i,r,v)\mathbf{e}_{r}\odot\mathbf{e}_{v}^{(l)}}}\end{array}
$$
显而易见，依然使用的是注意力机制
$$
\begin{array}{c}{{\beta(i,r,v)=\mathrm{softmax}\left((\mathbf{e}_{i}||\mathbf{e}_{r})^{T}\cdot(\mathbf{e}_{v}||\mathbf{e}_{r})\right)}}\\ {{=\displaystyle\frac{\exp\left((\mathbf{e}_{i}||\mathbf{e}_{r})^{T}\cdot(\mathbf{e}_{v}||\mathbf{e}_{r})\right)}{\sum\limits_{(v^{\prime},r)\in{\hat{\mathrm N}}(i)}\exp\left((\mathbf{e}_{i}||\mathbf{e}_{r})^{T}\cdot(\mathbf{e}_{v^{\prime}}||\mathbf{e}_{r})\right) },}}\end{array}
$$
然后将所有层的 representations 求和，得到全局表示 ${\mathbf z}^g_u$ and ${\mathbf z}^g_i$
$$
{\mathbf z}_{u}^{g}=\mathbf{e}_{u}^{(0)}+\cdot\cdot\cdot+\mathbf{e}_{u}^{(L^{\prime})},\quad {\mathbf z}_{i}^{g}=\mathbf{e}_{i}^{(0)}+\cdot\cdot\cdot+\mathbf{e}_{i}^{(L^{\prime})}
$$
**Global-level Cross-view Contrastive Optimization**

获得 global 和 local views 下的节点表示，首先将它们映射到计算对比损失的空间，与局部级对比损失计算相同
$$
\begin{array}{l}{{\mathbf {z}_{i}^{g}\_{\mathrm p}={W}^{(2)}\sigma\left({W}^{(1)}\mathbf {z}_{i}^{g}+b^{(1)}\right)+b^{(2)}}}\\ {{ {\mathbf {z}_{i}^{l}\_{\mathrm p}}={W}^{(2)}\sigma\left({ W}^{(1)}\mathbf ({\mathbf {z}}_{i}^{c}+{\mathbf {z}}_{i}^{s})+b^{(1)}\right)+b^{(2)}}}\end{array}
$$
从 global 视图和 local 视图计算的对比学习损失

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230912202557203.png" alt="image-20230912202557203" style="zoom:80%;" />

user 和 item 类似

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230912202527836.png" alt="image-20230912202527836" style="zoom:80%;" />

# Model Prediction

$$
\begin{array}{l c r}{{\mathbf{z}_{u}^{*}=\mathbf{z}_{u}^{g}||\mathbf{z}_{u}^{c}}}\\ {{\mathbf{z}_{i}^{*}=\mathbf{z}_{i}^{g}||(\mathbf{z}_{i}^{c}+\mathbf{z}_{i}^{s})}}\\ {{\mathbf{\hat{y}}(u,i)=\mathbf{z}_{u}^{*\top}\mathbf{z}_{i}^{*}}}\end{array}
$$

# Multi-task Training

$$
\mathcal{L}_{M C C L K}=\mathcal{L}_{\mathrm{BPR}}+\beta(\alpha\mathcal{L}^{l o c a l}+(1-\alpha)\mathcal{L}^{g l o b a l})+\lambda||\Theta||_{2}^{2}
$$

# EXPERIMENT

**How does MCCLK perform, compared to present models?**

作者将这种改进归因于以下几个方面：

1. 通过对比局部层面的协作和语义视图，MCCLK 能够更好地捕获协作和语义特征信息
2. 全局级对比机制保留了来自两级自判别的结构和特征信息，因此比仅建模全局结构的方法捕获了更全面的 MCCLK 信息

**Are the main components really working well?**

消融实验

**How do different hyper-parameter settings affect MCCLK?**

没啥好说的

**Is the self-supervised task really improving the representation learning?**

**继之前的对比学习工作**之后，我们采用 SVD 分解来投影将获得的项目嵌入到 2D 中并给出正则化的单数

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230912214219620.png" alt="image-20230912214219620" style="zoom:60%;" />