**老生常谈的问题**

1. 利用用户的个性化偏好来捕获多行为依赖关系
2. 处理目标行为监督信号稀疏导致推荐不足的问题

**出发点**

大多数现有方法只考虑用户和项目之间存在单一类型的行为，这使得它们无法从复杂的多重协作关系中提取有用的信息以提供足够的推荐，例如，用户可以在电子商务平台上以不同方式（例如点击、标记为最爱和购买）与商品进行交互

**现有模型的问题**

1. 目标行为数据稀疏性
2. 个性化行为多样性

**和常规的基于知识图感知的推荐不同的地方**

在 interaction graph 中多了一个 user-item 交互的关系属性 $G_{u}=({\mathcal{U}},\Theta,{\mathcal{V}})$，其中，$\Theta=\{\theta^{1},...,\theta^{k}\,,...,\theta^{K}\} $



# METHODOLOGY

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230829103800703.png" alt="image-20230829103800703" style="zoom:80%;" />

**主要由三个模块构成**

- multi-behavior learning module
- knowledge enhancement module
- joint learning module



## Multi-behavior Learning Module

**Multi-behavior Information Encoding and Aggregation**

为了充分利用多行为交互图，同时将高阶连接注入多路关系中
$$
v_{u}^{k},v_{i}^{k}=M u l t i E n c^{k}(G_{u},u,k)
$$
$v_{u}^{k},v_{i}^{k}$ 表示用户 u 和 项目 i 在行为 k 下的表示，$M u l t i E n c^{k}$ 表示一个可选择的多行为编码器，这里选择的是 LightGCN
$$
v_{u}^{k,(l+1)}=\sum_{i\in{\cal N}_{u}^{k}}v_{i}^{k,l};\qquad v_{i}^{k,(l+1)}=\sum_{u\in{\cal N}_{i}^{k}}v_{u}^{k,l}
$$
一般来说原始 user 和 item 的 embedding 定义为第 0 层，也就是 $v_{u}^{k,0},v_{i}^{k,0}$，作者在这里提出但行为表示可能会遇到数据稀疏问题，所以提出跨类型多行为嵌入聚合
$$
v_{u}=P R e L U((v_{u}^{0}||,...,||v_{u}^{l}||,...,||v_{u}^{L})\times W^{l})
$$

$$
v_{u}^{l}=\sigma(W^{u}\times m e a n(v_{u}^{1,l}\oplus,...,\oplus v_{u}^{k,l}\oplus,...,\oplus v_{u}^{K,l}))
$$

这里所谓的跨类型多行为嵌入聚合，首先将不同层的嵌入 concat 然后经过 trainable transformation matrix，把结果输入 PReLU 中得到，然后把不同 relation 的嵌入直加、mean 等系列操作得到最终表示

**Multi-behavior Contrastive Learning**

使用对比学习的原因是通过对比策略来最小化同一用户的多种行为之间的差异，最大化每个用户之间的差距

positive pair：用户 x 的不同行为 $\{v^k_x,v^{k^{\prime}}_x\}$

negative pair：用户 x 和用户 y 的行为 $\{v^k_x,v^{k^{\prime}}_y\}$
$$
{\mathcal{L}}_{M u l C L}=\sum_{k^{\prime}=1}^{K}\sum_{x\in{\mathcal{U}}}-l o g\frac{e x p(s(v_{x}^{k},v_{x}^{k^{\prime}})/\tau)}{\sum_{y\in{\mathcal{U}}}e x p(s(v_{x}^{k},v_{y}^{k^{\prime}})/\tau)}
$$


## Knowledge Enhancement Module

基于每个图嵌入表示，提出了一种从用户角度出发的 item score evaluation 的方法，通过保留低噪声 item 信息来构建 user-item 嵌入

**Item-side Information Encoding and Aggregation**

依然是最常用的注意力机制
$$
{\cal W}_{i,r,e}=\frac{f_{a t t}(v_{i},v_{r},v_{e})}{\sum_{e^{\prime}\in{\cal N}_{i}}f_{a t t}(v_{i},v_{r},v_{e^{\prime}})}
$$

$$
f_{a t t}(v_{i},v_{r},v_{e})=e x p\left[\sigma(W_{1}(v_{i}||v_{r}||v_{e}))+b_{1}\right]
$$

$$
v_{i}^{l+1}=\sigma(W_{2}(v_{i}^{l}+\sum_{e\in N_{i}}{\mathcal W}_{i,r,e}v_{e})+b_{2})
$$

**Knowledge-based Enhancement and Augmentation**

有意思的是与之前的直接用 transE/transR 的训练策略不同，在本文中作者使用了两种不同的方法来生成知识表示

TransR + TATEC
$$
f_{t d}(h,r,t)=-||M_{r}v_{h}+v_{r}-M_{r}v_{t}||_{2}^{2} 
$$

$$
f_{s m}(h,r,t)=v_{h}^{T}M_{r}v_{t}+v_{h}^{T}v_{r}+v_{t}^{T}v_{r}+v_{h}^{T}d D v_{t}
$$

基于以上两种知识图生成方式，会有两种 Loss Function
$$
\mathcal{L}_{t d}=\sum_{(h,r,t,t^{\prime})\in\mathcal{G}_{k1}}-l n\sigma(f_{t d}(h,r,t^{\prime})-f_{t d}(h,r,t))
$$

$$
\mathcal{L}_{s m}=\sum_{(h,r,t,t^{\prime})\in\mathcal{G}_{k2}}-l n\sigma(f_{s m}(h,r,t^{\prime})-f_{s m}(h,r,t))
$$

受之前工作的启发，计算“一致性”分数
$$
c=s(g_{1}(v_{i}),g_{1}^{\prime}(v_{i}));\qquad(g_{1},g_{1}^{\prime}\in{\cal G}_{k1}^{s u b}) 
$$

$$
c^{\prime}=s(g_{2}(v_{i}),g_{2}^{\prime}(v_{i}));\;\;\;\;\;\;\;(g_{2},g_{2}^{\prime}\in\mathcal{G}_{k2}^{s u b})
$$

公式中的 $
(g_{1},g_{1}^{\prime},g_{2},g_{2}^{\prime}) 
$ 是从 ${\mathcal G}_{k1},{\mathcal G}_{k2}$ 中通过设置不同的 seeds 得到的子图，$s(\sim)$ 是相似度函数计算 consistency parameter $c\in{\mathcal R}^{1\times M}$，并按照之前的知识图注意力机制生成子图的节点表示 $v_i$,同时作者也提到，稳定的节点有利于抵抗噪声，但是不足以反应用户的偏好，用户不一定对稳定性分数低的项目不感兴趣，所以作者尝试“从用户的角度”调整得分，并利用 socre 来指导 interaction graph 中 edges 的选择
$$
P_{u,i}=\sigma(v_{u}^{T}v_{i})\odot c 
$$

$$
\hat P_{u,i}=(1-M i n_{-}M a x(P_{u,i}))a+M i n_{-}M a x(P_{u,i})b
$$

根据这个概率，确定 interaction graph 中 edge 的保留，得到两个 user-item 子图 ${\mathcal G}_{u1},{\mathcal G}_{u2}$

**Knowledge-aware Contrastive Learning**

经过以上的步骤，分别得到了基于 KG 和基于交互的两会中不同的图 $({\mathcal G}_{u1},{\mathcal G}_{k1})$$({\mathcal G}_{u2},{\mathcal G}_{k2})$，然后就按照原始图的处理方式使用 Light 和 Graph-Attention 传播聚合信息，得到 embeddings
$$
{\mathcal{L}}_{K C L}=\sum_{x\in{\mathcal{U}}\cup{\mathcal{V}}}-l o g{\frac{e x p(s(v_{x}^{1},v_{x}^{2})/\tau)}{\sum_{x\in{\mathcal{U}}\cup{\mathcal{V}}}e x p(s(v_{x}^{1},v_{y}^{2})/\tau)}}
$$
通过多行为 CL 任务，KMCLR 可以学习更鲁棒的用户表示，捕获目标行为与其他行为之间的底层关系，并区分不同之间的个人偏好

## Joint Learning Module

$$
v_{u}=(1-\alpha)v_{u}^{m u l}+\alpha v_{u}^{k g},\qquad v_{i}=(1-\alpha)v_{i}^{m u l}+\alpha v_{i}^{k g}
$$

$$
{\mathcal{L}}_{B P R}=-\sum_{u\in{\mathcal{U}}}\sum_{i\in{\mathcal{N}}_{u}}\sum_{j\notin{\mathcal{N}}_{u}}I n\sigma(v_{u}^{T}v_{i}-v_{u}^{T}v_{j})
$$

# EXPERIMENTS

实验目的

- (RQ1)与其他推荐方法相比，KMCLR 的性能如何？
- (RQ2)KMCLR 中的不同模块对性能有何影响？
- (RQ3)KMCLR 如何有效缓解推荐的数据稀疏问题？
- (RQ4)不同的超参数设置如何影响性能？
- (RQ5)KMCLR 如何减轻额外信息引起的噪声问题？



## RQ1

（1）受益于多行为信息和附加 KG 的引入，模型可以充分捕获多方面的个性化信息作为信号来指导目标行为的推荐

（2）由于辅助自监督任务的引入，模型可以从更高层次的特征上挖掘个体偏好，清晰地区分个体之间的信息梯度

## Othrer

都是用实验来证明