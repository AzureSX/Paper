# INTRODUCTION

现有基于 GNN 的推荐系统的问题

Interaction dominate：用户-项目交互的监督信号将主导模型训练，这表明 KG 的信息几乎没有被编码在学习项目表示中

Knowledge overload： KG 包含许多与推荐无关的三元组，也就是噪声



# METHODOLOGY

## Classical GNN-based Recommender

**Graph Attention Network**
$$
\mathrm{h}_{\mathrm{{\cal{N}_i}}}={\sum}_{j\in{\mathrm{{\cal{N}}}i}}\,\alpha_{i j}\mathrm{h}_{j}
$$

$$
\alpha_{i j}=\frac{\exp(L e a k y R e L U(a_{b}^{T}[W_{b}{\bf h}_{i}||W_{b}{\bf h}_{j}]))}{\sum_{k\in{\cal N}_{i}}\exp(L e a k y R e L U(a_{b}^{T}[W_{b}{\bf h}_{i}||W_{b}{\bf h}_{k}]))}
$$

**Recommendation Loss**
$$
{\mathcal{L}}_{C F}(u,v^{+},v^{-})=-\log\sigma(y(u,v^{+})-y(u,v^{-}))
$$

## Knowledge-Adaptive Contrastive Learning

对比学习由是三个部分组成

1. 分别从交互图和知识图生成 adaptive view
2. 增强图上 user/item/entity 的关系感知结构编码
3. 对比学习任务，使 item 表示对两种视图共享和信息进行编码



**Adaptive Data Augmentation on Graph Structure**

这篇文章提出了一种新颖的增强方法，该方法首先在原始输入图上删除一定比例的边，得到 ${\mathcal G}_b，{\mathcal G}_k$ 的采样子集 $\tilde {\mathcal E}_b,\tilde {\mathcal E}_k$，然后从 $\tilde {\mathcal E}_b,\tilde {\mathcal E}_k$ 经由两个 learnable view generator 进一步删除不重要的边，generator 的任务是对 realvalued 重要性权重进行建模，计算每条边的采样概率  $p_e\in\{0,1\}$，1 会被保留，0 则会舍去

对于 interaction-graph，generator 定义如下
$$
w_{e}^{b}=M L P_{b}(\left[\mathrm{h}_{u}^{(0)}\vert\vert\mathrm{h}_{v}^{(0)}\ \right])
$$
由于 edge 的定义是 $e = (u,v)$ ，所以 $w_{e}^{b}$ 也表示 edge 的重要性，然后应用 Gumbel-Max 重新参数化的技巧
$$
p_{e}^{b}=\sigma((\log(\epsilon)-\log(1-\epsilon)+w_{e}^{b})/\tau_{b})
$$
最终得到增强后的图 ${\hat{\mathcal G}}_{b}$。相比于交互图，知识图上关系复杂且重要性差异巨大
$$
w_{e}^{k}=M L P_{k}(W_{r}^{k}({\mathbf e}_{h}^{(0)}||{\mathbf e}_{t}^{(0)}))
$$
$W_r^{k}$ 是关系 r 的变换矩阵，然后按照之前的方式生成 $p^k_e$
$$
p_{e}^{k}=\sigma((\log(\epsilon)-\log(1-\epsilon)+w_{e}^{k})/\tau_{k})
$$


**Relation-aware Graph Attention for Node Encoding**

对于交互图，采用的编码器为 $\rm GNN_{v1}$，也就是上面所列举出的传统的基于 GNN 的传播过程
$$
{\hat{\cal G}_b},{\hat {\bf h}_i^{(0)}}=>{\rm GNN_{v1}}=>{\hat {\bf h}_i}
$$
对于知识图，首先为每个 relation 和 entity 分配一个可学习的 embedding，然后用注意力机制
$$
\alpha_{h t}=\frac{\exp({{L e a k y R e L U}(a_{k}^{T}[W_{k}e_{h}||W_{r}}\mathrm{m}_{r(\langle h,t\rangle)}||W_{k}\mathrm{e}_{t}]))}{\sum_{j\in\cal N_{h}}\exp({{L e a k y R e L U}(a_{k}^{T}[W_{k}\mathrm{e}_{h}||W_{r}}\mathrm{m}_{r(\langle h,j\rangle)}||W_{k}\mathrm{e}_{j}]))}
$$
其中 ${\bf e}_h$ and ${\bf e}_t$ 是 entity embeddings，$r(\langle h,t\rangle)$ 表示头尾实体之间的关系，所以 $\mathrm{m}_{r(\langle h,t\rangle)}$ 表示 relation embedding，其余部分和 $\rm GNN_{v1}$ 相同
$$
{\hat{\cal G}_k}=>{\rm GNN_{v2}}=>{\hat {\bf e}_i}
$$
**Contrastive Learning Task**

由于交互和知识的视图处于不同的 representation space，所以将得到的两个嵌入 $(\hat{{\bf h}}_v,\hat{{\bf e}}_v)$ 输入到 MLPs 中，将它们映射到相同的向量空间中 $({\bf z}_v^b,{\bf z}_v^k)$，对比学习的策略是将同一 item 的两个视图视为 positive pair，不同 item 随机配对视为 negative pairs
$$
{\mathcal{L}}_{C L}(v)=-\log{\frac{\exp(s({\mathbf z}_{v}^{b},{\mathbf z}_{v}^{k})/\tau_{c l})}{\sum_{j\in\mathbb{N}\cup\{v\}}\exp(s({\mathbf z}_{v}^{b},{\mathbf z}_{j}^{k})/\tau_{c l})+\exp(s({\mathbf z}_{j}^{b},{\mathbf z}_{v}^{k})/\tau_{c l})}}
$$

## Model Prediction and Optimization

**The Overall Loss Function**

简单地连接来自推荐模块和对比模块的 user/item embeddings，但是知识图中仅包含 item，所以这里要为每一个 user 生成一个 trainable 的 $\hat {\bf e}_u$ 
$$
y(u,v)=(\mathbf{h}_{u}\vert\vert\hat{\mathbf{h}}_{u}\vert\vert\hat{\mathbf{e}}_{u})^{T}(\mathbf{h}_{v}\vert\vert\hat{\mathbf{h}}_{v}\vert\vert\hat{\mathbf{e}}_{v})
$$
然后引入一个 DistMult 来优化 KG 损失
$$
f(h,r,t)=\hat{\bf e}_{h}^{T}{R}_{r}\hat{\bf e}_{t}
$$
同样的，$R_r$ 是 relation r 的变换矩阵，$f$ 分数越低表明它更有可能是正确的
$$
{\mathcal{L}}_{K G}(h,r,t^{+},t^{-})=-\log\sigma(f(h,r,t^{-})-f(h,r,t^{+}))
$$
多任务联合训练
$$
\mathcal{L}=\mathcal{L}_{C F}+\lambda_{1}\mathcal{L}_{C L}+\lambda_{2}\mathcal{L}_{K G}
$$
虽然写了 $\lambda_1,\lambda_2$ 但实际训练过程中固定为 0.1 和 1



# EXPERIMENTS

**Baseline**

值得注意的是这篇文章它和 KGCL 做了对比，但是 batch_size 设置为 8192？

**为什么好**

1. 通过对交互图和KG分别进行编码，KACL可以更好地捕获用户偏好和项目知识信息（常规
2. 通过进一步对两张图进行知识自适应对比学习，KACL可以更好地对信息进行编码（动不动就加个自适应
3. 受益于对 KG 的正则化，KACL 可以有效地从实体和关系中收集更多信息信号（KG 正则化就是那个 DistMult?

**缓解数据稀疏性和冷启动**

分别做了实验

**消融实验**

![image-20230828194413567](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230828194413567.png)

分别删除了对比学习模块、自适应视图生成器、KG正则化