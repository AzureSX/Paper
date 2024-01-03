# ABSTRACT

数据噪声，它可能由于多种因素而产生，例如用户由于过度推荐热门商品而点击不相关的产品。直接聚合来自用户-项目交互图中所有交互边的信息可能会导致用户表示不准确，并且多跳嵌入传播可能会恶化噪声效应。

推荐数据的稀疏性和偏态分布会对有效的用户-项目交互建模产生负面影响

然后就说目前基于 random dropout 的方法还有缺陷



# METHODOLOGY

## **Local Collaborative Relation Learning**

遵循常见的 CF 范式，将 user/item 嵌入 $d$-dimensional latent space，具体来说就是为 user/item 分别生成 大小为 $\mathbb R^d$ 的 embedding vectors $\mathbf e_i$ and $\mathbf e_j$ ，定义 $\mathbf E^u \in \mathbb R^{I\times d}$ and $\mathbf E^v \in \mathbb R^{J\times d}$，接下来使用 LightGCN 聚合信息
$$
\mathbf{z}_{i}^{\left(u\right)}=\bar{\mathcal{A}}_{i,*}\cdot\mathbf{E}^{\left(v\right)},\;\;\;\;\mathbf{z}_{j}^{\left(v\right)}=\bar{\mathcal{A}}_{*,j}\cdot\mathbf{E}^{\left(u\right)}
$$

$$
\bar{\mathcal{A}}=\operatorname{D}_{(u)}^{-1/2}\cdot\mathcal{A}\cdot\operatorname{D}_{(v)}^{-1/2},\quad\bar{\mathcal{A}}_{i,j}=\frac{\mathcal{A}_{i,j}}{\sqrt{|\mathcal{N}_{i}|\cdot|\mathcal{N}_{j}|}}
$$

将 user $u_i$ 和 item $v_j$ 在第 $𝑙$ 层的嵌入分别表示为 $\mathbf{e}^{(u)}_{i,l}$ and $\mathbf{e}^{(v)}_{j,l}$
$$
\mathbf{e}_{i,l}^{(u)}=\mathbf{z}_{i,l}^{(u)}+\mathbf{e}_{i,l-1}^{(u)},\;\;\;\mathbf{e}_{j,l}^{(v)}=\mathbf{z}_{j,l}^{(v)}+\mathbf{e}_{j,l-1}^{(v)}
$$
且 final embedding 由各层求和得到
$$
\mathbf{e}_{i}^{(u)}=\sum_{l=0}^{L}\mathbf{e}_{i,l}^{(u)},\;\;\mathbf{e}_{j}^{(v)}=\sum_{l=0}^{L}\mathbf{e}_{j,l}^{(v)},\;\;\hat{y}_{i,j}=\mathbf{e}_{i}^{(u)\top}{\mathbf{e}_{j}^{(v)}}
$$

## **Adaptive View Generators for Graph Contrastive Learning**

Dual-View GCL Paradigm

现存方法，以特定方式生成视图，例如随机删除边、节点或构造超图。本文提出使用两个不同的视图生成器，从不同的角度增

强用户-项目图，具体来说，采用图生成模型（graph generative model）和图去噪模型（graph denoising model）

作为两个视图生成器，图生成模型负责根据图分布重建视图，而图去噪模型则利用图的拓扑信息去除 user-item graph 中

的噪声并生成噪声更少的新视图，根据现有的自监督 CF 范式，使用节点 self-discrimination 来生成正负对
$$
{\cal L}_{s s l}^{u s e r}=\sum_{u_{i}\in{\cal U}}-\log\frac{\exp(s({\bf e}_{i}^{\prime},{\bf e}_{i}^{\prime\prime})/\tau)}{\sum_{u_{i^{\prime}}\in{\cal U}}\exp(s({\bf e}_{i}^{\prime},{\bf e}_{i^{\prime}}^{\prime\prime}/\tau)}
$$

$$
{\mathcal{L}}_{s s l}={\mathcal{L}}_{s s l}^{u s e r}+{\mathcal{L}}_{s s l}^{i t e m}
$$



**Graph Generative Model as View Generator**

基于学习的图生成模型为视图生成器提供了一个有前途的解决方案，本文采用广泛使用的变分图自动编码器（VGAE）作为生成

模型。此外，VGAE 比其他当前流行的生成模型（generative adversarial networks and diffusion models）相

对更容易训练并且速度更快。

首先，利用多层 GCN 作为 encoder 获得 graph embeddings，然后再利用两个 MLP 分别导出 graph embeddings 的平均值（mean value）和标准差（standard deviation），使用另一个 MLP 作为 decoder，对输入的平均值和带有高斯噪声的标准差进行解码，生成新的图。VGAE 损失定义为
$$
{\mathcal{L}}_{g e n}={\mathcal{L}}_{k l}+{\mathcal{L}}_{d i s}
$$
${\mathcal{L}}_{k l},{\mathcal{L}}_{d i s}$ 分别表示 node embeddings 和 standard Gaussian distribution 之间的 KL 散度，以及生成图和原

始图之间的差异



**Graph Denoising Model as View Generator**

沿噪声边缘聚合的消息会降低节点嵌入的质量。因此，对于第二个视图生成器，本文的目标是生成一个去噪视图，可以增强模

型针对噪声数据的性能，对于第二个视图生成器，本文的目标是生成一个去噪视图，可以增强模型针对噪声数据的性能。背后

的主要概念是使用参数化网络（parameterized network）主动过滤掉输入图中的噪声边缘

首先定义一个 binary matrix $\mathbf{M}^l\in 0,1^{|{\mathcal V}|\times|{\mathcal V}|}$，matrix 中的元素表示节点 $u_i$ 和 $v_j$ 之间是否存在边（0表示存

在噪声，为什么？？？），所得子图为 ${\mathbf A}^l={\mathbf A}\odot{\mathbf M}^l$，其原理是 penalize the number of non-zero entries 

in ${\mathbf M}^l$ of different layers
$$
\sum_{l=1}^{L}||{\bf M}^{l}||_{0}=\sum_{l=1}^{L}\sum_{(u,v)\in\varepsilon}\mathbb {I}[m_{i,j}^{l}\neq0]
$$
$\mathbb I[·]$ 是一个 indicator 函数，when $\mathbb I[True]=1$ and $\mathbb I[False]=0$，$||·||_0$ 表示 $l_0$ norm，简而言之就是计

算 L 层矩阵 $\bf M$ 的非零 edge 的和。然而，由于其 combinatorial and non-differentiability nature，优化

这种惩罚在计算上是困难的。因此作者从 Bernoulli distribution parameterized 得出每个 binary number 
$$
m_{i,j}^{l}\sim\mathrm{Bern}(\pi_{i,j}^{l})
$$
$\pi_{i,j}^{l}$ 描述了 edge $(u,v)$ 的质量，为了使用梯度方法有效地优化子图，采用重新参数化技巧，并将从伯努利分布绘制

的 binary entries $m_{i,j}^{l}$ 放宽为参数 $\alpha^l_{i,j}\in {\mathbb R}$ 和独立随机变量 $\varepsilon^{l}$ 的确定性函数 $g$
$$
m_{i,j}^{l}=g(\alpha_{i,j}^{l},\varepsilon^{l})
$$
基于以上操作，作者设计了一个去噪层来学习控制是否去除边缘 $(u,v)$ 的参数 $\alpha_{i,j}^{l}$，对于第 $l$ 层 GNN，计算 user $u$ 和其交互的 item $v$ 用如下公式计算
$$
\alpha_{i,j}^{l}=f_{\theta^{l}}^{l}(\mathbf{e}_{i}^{l},\mathbf{e}_{j}^{l})
$$
$f_{\theta^{l}}^{l}$ 是由 $\theta^l$ 参数化的 MLP ，还利用了一个 concrete distribution along 和 hard sigmoid 函数，所以之前的

公式可以重构为
$$
{\mathcal{L}}_{c}=\sum_{l=1}^{L}\sum_{(u_{i},v_{j})\in\varepsilon}\left(1-\mathbb{P}_{\sigma}(s_{i,j}^{l})\left(0|\theta^{l}\right)\right)
$$
$\mathbb{P}_{\sigma}(s_{i,j}^{l})$ 是 $\sigma(s_{i,j}^{l}) $ 的累积分布函数 (CDF)，$\sigma(·)$ 扩展了 $s_{i,j}^{l}$ 的范围，$s_{i,j}^{l}$ 取自 a binary concrete 

distribution，且 $\alpha_{i,j}^{l}$ 参数化 location

##  Learning Task-aware View Generators

引入常用的BPR损失
$$
{\mathcal{L}}_{b p r}=\sum_{(u,i,j)\in O}-\log\sigma({\hat{y}}_{u i}-{\hat{y}}_{u j})
$$
为了训练图生成模型，使用 VGAE 编码器编码的节点嵌入来计算 BPR 损失
$$
{\mathcal{L}}_{g e n}={\mathcal{L}}_{k l}+{\mathcal{L}}_{d i s}+{\mathcal{L}}_{b p r}^{g e n}+\lambda_{2}\vert\vert\Theta\vert\vert_{\mathrm{F}}^{2}
$$
为了训练图去噪模型，使用去噪神经网络获得的节点嵌入来计算 BPR 损失
$$
{\mathcal{L}}_{d e n}={\mathcal{L}}_{c}+{\mathcal{L}}_{b p r}^{d e n}+\lambda_{2}||\Theta||_{\mathrm{F}}^{2}
$$

## Model Training

模型的训练由两部分组成。在上层部分，采用多任务训练策略来联合优化经典推荐任务和自监督学习任务
$$
\mathcal{L}_{u p p e r}=\mathcal{L}_{b p r}+\lambda_{1}\mathcal{L}_{s s l}+\lambda_{2}||\Theta||_{\mathrm{F}}^{2}
$$
训练的较低级别部分涉及基于方程式优化生成和去噪视图生成器
$$
\mathcal{L}_{l o w e r}=\mathcal{L}_{g e n}+\mathcal{L}_{d e n}
$$




# 总结

节点邻居信息传播聚合方式仍然是沿用 LightGCN

在经过 GCN 聚合之后获得 embeddings 后，利用 VGAE 获得生成图

利用去噪模块，获得去噪视图

