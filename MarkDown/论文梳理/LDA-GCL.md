**Abstract**

最近，图神经网络（GNN）在推荐领域取得了显着的成功。为了减少数据稀疏性的影响，基于 GNN 的 CF 方法采用图对比学习（GCL）来提高性能。大多数 GCL 方法由数据增强和对比损失组成（InfoNCE）。 GCL 方法通过手工制作的图形增强来构造对比对，并最大化同一节点的不同视图与其他节点的视图之间的一致性，这被称为 **InfoMax** 原理

然而，不恰当的数据增强会阻碍 GCL 的性能。 **InfoMin** 原则，即良好的视图集共享最少的信息，并为设计更好的数据增强提供指导



# INTRODUCTION

尽管 GNN 模型取得了显着的成功，但它们仍然面临数据稀疏问题。为了克服这个困难，引入了自监督方式的图对比学习（GCL）来提高推荐性能。GCL 有两个重要的组成部分：数据增强（data augmentation）和对比损失（contrastive loss）。对于数据增强，过去的 GCL 方法通过 edge-dropping 生成手工图形增强，数据增强后，GCL 使用图神经网络模型来获取多个视图上的节点表示。对比损失（InfoNCE）利用互信息最大化原则（InfoMax），旨在最大化不同增强图中节点表示之间的对应关系

然而，不适当的数据增强可能会阻碍对比学习的性能。如何找到合适的数据增强是一个有前途的研究问题。 Tian等人调查了对比学习（CL）中如何形成良好观点的研究问题。受信息瓶颈（IB）的启发（2000年的文章？），他们提出了 **InfoMin** 原则，即良好的视图集共享在下游任务中表现良好所需的最少信息（the good set of views shares the minimal information necessary to perform well at the downstream task）。他们发现更强的数据增强确实会导致相互信息减少并改善下游任务。 InfoMin 原则为我们在推荐中找到GCL的最佳数据增强提供了指导



# Preliminary

**GNN-based Collaborative Filtering**
$$
z_{w}^{l}=f_{\mathrm{aggregate}}(\left\{{{\mathcal{z}}_{v}^{l-1}\mid v\in{\mathcal{N}}_{w}\cup\left\{w\right\}}\right\} )
$$

$$
z_{w}\,=\,f_{\mathrm{update}}\,\left(\left[z_{w}^{0},z_{w}^{1},\ldots,z_{w}^{L}\right]\right)
$$

$$
Z^{l+1}=\left({\bf D}^{-\frac{1}{2}}{\bf A}{\bf D}^{-\frac{1}{2}}\right)Z^{l},Z=\frac{1}{L+1}(Z^{0}+Z^{1}+\cdot\cdot\cdot+Z^{L})
$$

**Graph Contrastive Learning in Recommendation**
$$
{\mathcal{L}}_{\mathrm{NCE}}^{\mathcal{U}}=\sum_{u\in{\mathcal{U}}}-\log\frac{\exp\left(s i m\left(\mathbf{z}_{u}^{\prime},\mathbf{z}_{u}^{\prime\prime}\right)/\tau\right)}{\sum_{v\in{\mathcal{U}}}\exp\left(s i m\left(\mathbf{z}_{u}^{\prime},\mathbf{z}_{v}^{\prime\prime}\right)/\tau\right)}
$$

$$
{\mathcal{L}}_{\mathrm{NCE}}={\mathcal{L}}_{\mathrm{NCE}}^{\mathcal U}+{\mathcal{L}}_{\mathrm{NCE}}^{\mathcal I}
$$

推荐系统中的对比学习通常采用联合学习（joint learning）策略来训练模型，而不是预训练和微调策略。换句话说，pretext 任务和 downstream 任务都是联合优化的。 Wu 等人证明联合训练将取得更好的性能，两种任务相互增强



# METHODOLOGY

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230914154009695.png" alt="image-20230914154009695" style="zoom:70%;" />



**Graph Data Augmentation With Edge Operating**

推荐系统中现有的 data augmentation 通常是 edge-dropping 其他 node dropping 和 random walk dropping 也可以视为不同的边丢弃策略
$$
s_{1}(G)=\mathbf{A}_{1}=\mathbf{A}\odot\mathbf{M}_{1},\quad s_{2}(G)=\mathbf{A}_{2}=\mathbf{A}\odot\mathbf{M}_{2}
$$
然而，唯一的边缘丢弃策略将会遇到数据稀疏问题，本文提出了推荐系统中的新数据增强 **edge-operating** including

both **edge-adding** and **edge-dropping**

与图上自监督的边缘扰动相比，推荐系统中的边缘操作面临一些挑战

i) 从 $\mathbf A$ 中随机采样边的复杂度是 ${\mathcal O((|V|)^2)}$，这在具有百万级用户和项目的大规模推荐系统中是不可接受的

ii) 向 $\mathbf A$ 随机添加边会引入噪声

因此，本文提出首先构建候选边缘（edge candidates），然后从这些候选边缘中进行采样。先训练一个 GNN 模型来预测用户 $u$ 的 item 偏好 (Edge Suggestion)。然以选择 top-$K_u$ 个 item 作为候选，$K_u$ 是 user 在交互矩阵上的度。添加边后，最终候选边由 original edges $\mathcal E_0$ 和 suggested edges ${\mathcal{E}}_{1}\left(|{\mathcal{E}}_{0}|\,=\,|{\mathcal{E}}_{1}|\right) $ 组成



**Learning Data Augmentation**

在引入 edge-operating augmentation 之后，本文进一步提出使用 earnable edge operator model $t$ 来生成信息数据增强而不是随机采样，使用多层感知机（MLP）来学习每个边缘候选 $e_{u,i}$ 的权重
$$
w_{u,i}=\mathrm{MLP}\left(\left[z_{u}\odot z_{i}\right]\left|\right|\mathbb 1_{{\mathcal{E}}}\left(e_{u,i}\right)\right)
$$
$\mathbb 1_{{\mathcal{E}}}\left(e_{u,i}\right)$ 表示边 $e_{u,i}$ 属于原始边还是添加边，再使用 Gumbel-Max 再参数化获得边 $e_{u,i}$ 的概率 $p_{u,i}$
$$
p_{u,i}=\mathrm{signuoid}\bigl(\frac{(\mathrm{log~}\delta-\mathrm{log}(1-\delta)+\omega_{u,i})}{\tau}\bigr)
$$
这种边缘学习风格也被用于 GNN 的参数化解释和对抗性攻击（parameterized explanations and adversarial
attacks），然后使用 $p_{u,i}$ 构造 augmented graph 
$$
t(G)={\mathbf A^{\prime}}=\left(\begin{array}{c c}{{\mathbf 0}}&{{\mathbf P}}\\ {{\mathbf P^{\top}}}&{{\mathbf 0}}\end{array}\right)
$$
LightGCN 应用于具有邻接矩阵 $\mathbf A$ 的原始图 $G_1 = G$ 和具有邻接矩阵 $\mathbf A^{\prime}$ 的增强图 $G_2 = t(G)$ ，通过LightGCN 得到 embeddings $Z_1$ 和 $Z_2$



**Objective Function**

受到 graph contrastive learning 相关工作的启发，这里使用 adversarial loss function 来寻找良好的图增强来增强 GCL 的推荐
$$
\operatorname*{min}_{t}\,\,\lambda_{t}I(f(G);f(t(G)))+{\cal L}\big(f(t(G)),y)
$$

$$
\operatorname*{max}_{f}~I(f(G);f(t(G)))-\mathcal{L}(f(G),y),
$$

$I(X_1;X_2)$：是两个随机变量 $X_1$ 和 $X_2$ 之间的 mutual information

$t$：data augmentation learner

$f$：GNN encoder (i.e., LightGCN)

$\mathcal L$：BPR loss function

$\lambda_{t}$：the influence of $I$ for $t$

为了评估互信息（MI），这里采用常见的 InfoNCE，所以以上公式中的 $I(f(G),f(t(G)) $ 可以替换为
$$
I(f(G),f(t(G))\to-{\mathcal{L}}_{\mathrm{NCE}}={\frac{1}{B}}\sum_{i=1}^{B}\log{\frac{\exp{(s i m\left(z_{i,1},z_{i,2}\right))}}{\sum^B_{i^{\prime}=1,i^{\prime}\neq i}\exp{(s i m\left(z_{i,1},z_{i^{\prime},2}\right))}}}
$$
由于是 min-max optimize problem，这里使用对抗训练中使用的迭代训练方法

当固定 $t$ 时
$$
{\mathcal{L}}_{f}={\mathcal{L}}_{\mathrm{BPR}}(f(G),y)+\lambda_{s s l}{\mathcal{L}}_{\mathrm{NCE}}\ (f(G),f(t(G)))+\lambda_{r e g}\|f\|_{2}^{2}
$$
当固定 $f$ 时
$$
{\mathcal{L}}_{t}={\mathcal{L}}_{\mathrm{BPR}}(f(t(G)),y)-\lambda_{2}{\mathcal{L}}_{\mathrm{NCE}}\ (f(G),f(t(G)))+\lambda_{r e g}\|t\|_{2}^{2}
$$


**Training LDA-GCL**

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230914194351861.png" alt="image-20230914194351861" style="zoom:80%;" />

**Input**

- 原始图 $G(\mathcal{U,I,E})$
- Pre-trained GNN encoder $f_0$（用于生成添加边）
- GNN encoder $f$ （用于生成 $Z$）
- Edge operator model $t$ （MLP用于生成边的权值）
- Epoch $T$

**Output: **Node representation $Z$

1. 先从预训练模型 $f_0$ 中获取要添加的边 $\mathcal E_1$
2. 候选边集合即为原始边和添加边和集合 $\mathcal E_2=\mathcal E_0+\mathcal E_1$
3. 初始化 $t$ 和 $f$ 的参数
4. 进入单个 epoch
   1. 对于每个 mini-batch 的交互 $B$
      1. 获取 user set $U$，item set $I$ 
      2. 固定 $f$，解冻 $t$
      3. 将 $t$ 应用于 $\mathcal E_2$ 获得增强图 $t(G)$ 同时使用 $f$ 从原始/增强图获得 final embeddings $Z_1/Z_2$ 
      4. 计算 $\mathcal L_t$ 的损失，反向传播，更新 $t$ 的参数
      5. 固定 $t$，解冻 $f$
      6. 同步骤 3
      7. 计算 $\mathcal L_f$ 的损失，反向传播，更新 $f$ 的参数
      8. 如果 $Z_1$ 满足 early stopping 的条件
      9. 停止算法，返回 best GNN encoder $f_{opt}$
5. 返回 $Z=f_{opt}(G)$



# **什么是迭代训练方法**

min-max 优化是对抗性训练中常见的方法，特别是在生成对抗网络（GANs）中。在 GANs 中，有两个网络：生成器和判别器。生成器的目标是生成与真实数据相似的数据，而判别器则试图区分真实数据和生成的数据。这两个网络在一个最小-最大博弈中进行训练，生成器试图最小化判别器区分虚假数据和真实数据的能力，而判别器试图最大化在它们之间进行区分的准确性

迭代训练方法涉及反复更新生成器（generator）和判别器（discriminator）

1. **初始化生成器和判别器**：首先使用随机权重初始化生成器和判别器网络
2. **生成虚假(增强)数据**：使用当前的生成器生成一批虚假(增强)数据
3. **训练判别器（最大化）**：
   - 计算判别器的损失。通常，这涉及比较判别器对真实数据和虚假数据的预测与它们各自的标签（真实数据标签为1，虚假数据标签为0）
   - 通过判别器网络反向传播损失，并使用随机梯度下降（SGD）或 Adam 等优化器更新其权重。这一步的目标是最大化判别器区分真实数据和虚假数据的能力
4. **再次生成虚假(增强)数据**：使用更新后的生成器生成新一批虚假(增强)数据
5. **训练生成器（最小化）**：
   - 计算生成器的损失。这个损失基于判别器对生成器生成的虚假数据的输出
   - 通过生成器网络反向传播损失并更新其权重。这里的目标是最小化判别器区分虚假数据和真实数据的能力
6. **重复**：持续执行步骤3至5，直到达到一定的迭代次数或收敛为止
7. **评估**：定期在验证数据集上评估生成器和判别器的性能，或者使用其他合适的指标进行评估
8. **重复和调整超参数**：根据需要调整超参数，如学习率、批大小和网络架构，然后重复训练过程，直到达到所需的结果

通过在这种最小-最大方式下迭代训练生成器和判别器，生成器变得更擅长生成逼真的数据，而判别器变得更擅长区分真实数据和虚假数据。这个过程会持续下去，直到达到平衡点，理想情况下生成器生成的数据与真实数据无法区分，判别器在区分真实和虚假数据方面达到50%的准确性


$$
E_{\mathrm{neg}}(u^{*})\equiv\left\{(u^{*},v)\mid(u^{*},v)\in E_{\mathrm{neg}}\right\}
$$
