# **现存问题**

仍然是稀疏性、长尾分布、目前的基于图的 CL 方法不能放直接应用于基于 KG 的推荐



# 说明问题

展示不同数据的长尾分布问题



# **Methodology**

## Framework

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230823160045023.png" alt="image-20230823160045023" style="zoom:50%;" />

首先在原始 CKG 中进行正常的 BPR Loss 操作（包含 KG 损失操作），再把根据 Dropout 方法生成 2 个 Augmented CKG 和由原始图生成的 users/entities 传入 Projection Head Layer 将学习到的嵌入表示映射到稀疏的高维空间，在此之后分别进行 user/entity/user-item 级别的对比，最后汇总损失

CKG：${\mathcal G} = \{(h,r,t)\:|h,t\in{\mathcal{E}}\cup{\mathcal{U}},r\in{\mathcal{R}}\}$ ，$\mathcal{E}$ 表示实体集合，$\mathcal R$ 表示关系集合，其中 Interact = 1 被视作一种边



## Graph Augmentation

仍然是使用 Dropout 的方法，分别是 Node Dropout 和 Edge Dropout

**Dropout**

图的每个节点及其连接边都会以概率 ρ 被丢弃。这种增强方法有助于识别图中更有影响力的节点

**Edge Dropout**

图中以概率 ρ 丢弃的边。这种增强方法有助于捕获图中更有影响力的子结构

## Representation Learning Layer

表示学习层分为三个部分

- attentive embedding propagation layer
- high-order propagation
- supervised signal integration

***

**Attentive Embedding Propagation Layers**

**Information Propagation**
$$
v_{{\cal H}}=\sum_{(h,r,t)\in{\cal H}}\xi\left(h,r,t\right)v_{t}
$$
**Knowledge-aware Attention**
$$
\xi\left(h,r,t\right)=\left(W_{r}v_{t}\right)^{T}\operatorname{tanh}\left(\left(W_{r}v_{h}+v_{r}\right)\right)
$$

$$
\xi\left(h,r,t\right)=\frac{\exp\left(\xi\left(h,r,t\right)\right)}{\sum(h,r^{\prime},t^{\prime})\in{\cal H}}\exp\left(\xi\left(h,r^{\prime},t^{\prime}\right)\right)
$$
**Information Aggregation**
$$
v_{h}^{(1)}=f\left(v_{h},v_{{\cal H}}\right)=L e a k y R e l u\left(W^{'}\left(v_{h}+v_{{\cal H}}\right)\right)+L e a k y R e l u\left(W^{'^{\prime}}\left(v_{h}\odot v_{{\cal H}}\right)\right)
$$

***

**High-Order Propagation**
$$
v_{h}^{(l)}=f\left(v_{h}^{(l-1)},v_{\mathcal H}^{(l-1)}\right)
$$

$$
v_{u}^{*}=v_{u}^{(0)}\parallel\cdot\cdot\cdot\parallel v_{u}^{(L)},\ v_{i}^{*}=v_{i}^{(0)}\parallel\cdot\cdot\cdot\parallel v_{i}^{(L)},v_{e}^{*}=v_{e}^{(0)}\parallel\cdot\cdot\cdot\parallel v_{e}^{(L)}
$$

***

**Supervised Signal Integration**
$$
v=M L P\left(v^{*}\right)=W_{2}\sigma\left(W_{1}v^{*}\right)
$$
由于学习到的嵌入表示通常是密集的，因此很难在 CL 的密集空间中更均匀地划分它们。有必要将学习到的嵌入表示映射到稀疏的高维空间。为此，在表示层中引入投影头层以进一步整合监督信号。投影头层的实现是两层多层感知器（MLP）



## Multi-level Contrastive Learning

$$
{\cal L}_{c l}={\cal L}_{u s e r-l e v e l}+{\cal L}_{e n t i t y-l e v e l}+{\cal L}_{u s e r-i t e m-l e v e l}
$$

$$
L_{c l}=\sum_{i\in{\cal B}}-\log\frac{\exp\left(z_{i}^{'T}z_{i}^{'^{\prime}}\right)\tau}{\sum_{j\in{\cal B}}\exp\left(z_{i}^{'T}z_{j}^{'^{\prime}}\right)\tau} )
$$

**User-Level Contrastive Learning**

用户级对比学习可以在嵌入空间中缩短具有相似兴趣的用户之间的距离并拉宽具有不同兴趣的用户之间的距离。通过用户级的CL，可以增加具有相似兴趣的用户之间的区分程度，使得用户节点的表示更加独立。从而可以挖掘用户更详细的兴趣，提高推荐的准确性

**Entity-Level Contrastive Learning**

实体级对比学习可以增加不同实体之间的区分度，实体节点表示在嵌入空间中的分布变得更加均匀，缓解实体节点的长尾问题

**User-Item-Level Contrastive Learning**

User-Item-Level Contrastive Learning 是为了让 CL 任务与推荐任务更加兼容。在嵌入空间中，缩短了用户与其交互项之间的距离，从而最大化了交互项与用户真正感兴趣的项之间的相似度。这样就可以挖掘出用户真正感兴趣的项，而不是推荐高曝光度的单品



## Joint Learning Paradigm

$$
{\cal L}_{j o i n t}={\cal L}_{k g}+{\cal L}_{b p r}+\lambda_{1}{\cal L}_{c l}+\lambda_{2}\left\|\theta\right\|_{2}^{2}
$$

$$
{\cal L}_{k g}=\sum_{(h,r,t)\in{\cal G},(h,r,t^{\prime})\not\in{\cal G}}-\ln\sigma\left(S c o r e(h,r,t^{\prime})-S c o r e\left(h,r,t\right)\right)
$$

$$
{\cal L}_{b p r}=\sum_{(u,i)\in{\cal R}^{+},(u,j)\in{\cal R}^{-}}-\mathrm{ln}\,\sigma\,\bigl(\hat{y}\left(u,i\right)-\hat{y}\left(u,j\right)\bigr)
$$

$$
\hat{y}\left(u,i\right)=v_{u}^{\ast T}v_{i}^{\ast}
$$



# Experiments

基于 Recbole library 实现了 ML-KGCL

## Compared with SOTA 

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230823153310508.png" alt="image-20230823153310508" style="zoom:50%;" />

## Ablation Experiment

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230823160232035.png" alt="image-20230823160232035" style="zoom:50%;" />

**Effect of Projection Head Layer**

利用投影头层整合监督信号可以显着提高推荐效果，验证了投影头层的有效性,在取消 PH 模块之后 ML-KGCL 的性能还不如KGAT,这一结果表明，在稠密空间中均匀地表示数据分布是很困难的。并且由于数据分布过于密集，会降低推荐效果。它还说明了将嵌入映射到高维稀疏空间对于 CL 的重要性

## Alleviate the Long-Tail Issue

$$
R e c a l l=\frac{1}{\mathcal{N}}\sum_{u=1}^{\mathcal{N}}\frac{\sum_{{\mathcal{M}}=1}^{10}\left|\left(l_{r e c}^{u}\right)^{\mathcal{M}}\cap l_{i n t e r a c t}^{u}\right|}{|l_{i n t e r a c t}^{u}|}=\sum_{\mathcal{M}=1}^{10}R e c a l l^{\mathcal{M}}
$$

$l_{rec}^{u}$ 和 $l_{interact}^{u}$ 分别表示推荐给用户 u 的项目，以及与用户 u 交互的所有项目，${\mathcal N}$ 表示用户的数量，$Recall^{\cal M}$ 表示第 m 组的贡献，由这组公式可以衡量不同流行程度的项目对于推荐的贡献

为了说明基于自监督学习的 ML-KGCL 为什么推荐效果优于 KGAT，利用 t-SNE 数据降维方法来降低表示的维度并展示，从图中可以看到，与 KGAT 模型相比，ML-KGCL 在每个级别上的嵌入表示分布更加均匀