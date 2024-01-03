# ABSTRACT

先引出 SGL，但是提出 SGL 直接使用的 SimCLR 的对比学习框架，该框架“可能”无法直接适用于推荐系统的场景，并且没有充分考虑 user-item 交互的不确定性（这句话在说什么？

# INTRODUCTION

GCN 为解决用户-项目二部图稀疏问题和交互式建模问题提供了最先进、最高效的解决方案。然而，GCN 学习到的表示很容易偏向于高频项或用户，也很容易受到交互噪声的影响。SGL 为解决这个问题诞生，通过增强输入数据来使用未标记的数据空间，从而实现下游任务的显着改进

虽然 SGL 取得了一定程度的效果提升，但作者认为 SGL 的对比学习框架可能不太适合推荐任务。SGL使用 simCLR 的对比学习框架作为自监督学习的范例，它将输入数据的增广样本视为正样本，并将同一批次中的其他样本视为负样本。由于计算机视觉任务中样本的多样性，这种设计是合理的，有助于挖掘硬负样本，从而提高学习表示的质量。然而，在推荐系统中，比较的目标是用户或物品节点，这意味着批次中很大概率存在相似的用户或物品。将这些样本视为负样本，相似的用户或物品在表示空间中会变得更远，这违背了推荐系统的优化目的，影响了 GCN 学习到的最终表示，降低了推荐系统的性能

为了解决上述局限性，本文提出了一种称为监督对比学习（SCL）的学习范式用于推荐

# PRELIMINARIES

## PROBLEM STATEMENT

传统的 user-item interaction graph
$$
{\mathcal{G}}=(\mathcal{U},\mathcal{V},\mathcal{E})
$$

## GRAPH CONVOLUTIONAL NETWORK

GCN 的核心是聚合二部图 G 上每个节点的域表示，并通过考虑高阶邻居的信息来更新每个节点的表示，最终得到有效的表示
$$
u_{i}^{\,\,\,(l)}\,=\,\delta\!\left(W^{(l)}[u^{(l-1)}{}_{i}|A(v_{j}\in\mathcal{N}\left(u_{i}\right)]\right)
$$

$$
v_{j}^{\,\,\,(l)}\,=\,\delta\!\left(W^{(l)}[v^{(l-1)}{}_{j}|A(u_{i}\in\mathcal{N}\left(v_{j}\right)]\right)
$$

得到的第 l 层的表示对应于节点第 l 跳邻居的信息聚合。对于不同层的信息，有一个 readout 函数来生成最终的节点表示，以 user side 为例
$$
u_{i}=f_{\mathrm{readout}}\left(\left\{u_{i}^{(l)}\mid l=\left[0,\cdots,L\right]\right\}\right)
$$
在获得每个节点的最终表示后，预测层可以用于计算某个用户 i 对项目 j 的偏好。为了保证 GCN 的计算效率，目前广泛使用向量内积作为预测层
$$
\hat{y}_{i j}\,=\,u_{i}^{\ T}v_{j}
$$
在优化模型参数时，可以使用二元交叉熵（BCE）和均方误差（MSE）等逐点损失函数，但GCN由于其性能更好，通常使用 BPR 损失
$$
{\cal L}_{B P R}=-\sum_{u=1}^{M}\sum_{i\in{\cal N}_{u}}\sum_{j\notin{\cal N}_{u}}\mathrm{ln}\sigma\left(\hat{y}_{u i}-\hat{y}_{u j}\right)+\lambda{\cal L}_{2}
$$

## BACKGROUND CONTRASTIVE LEARNING

对比学习的框架如图所示。它最大化相同数据生成的增强样本之间的相似性，并最小化与同一批次中其他样本的相似性来学习表示

![image-20230908151904417](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230908151904417.png)

该框架包括四个组件：数据增强(data augmentation)、编码器(encoder)、投影头(encoder)和对比度损失函数(contrast loss function)

数据增强模块通过一些随机数据增强方法从原始数据 $X_1$ 生成相关视图，表示为 $X^1_1$ 和 $X^2_1$。在二部图中，常用的数据增强方法包括节点删除(node drop)、边缘删除(edge drop)和节点复制(node replication)。不同的数据增强方法对比较学习的性能有很大影响

编码器从增强视图生成表示向量 $H^1_1$ 和 $H^2_1$ 。在用户-项目二部图问题中，编码器是 GCN 网络

投影头的目的是将表示投影到对比度损失空间中。投影头一般是带有隐藏层的MLP。许多关于对比学习的实验证明，投影头有利于提高对比学习的性能

称为 InfoNCE 的对比度损失函数旨在最小化对比度损失空间中同一样本生成的增强视图的距离，同时最大化该空间中同一批次中的其他样本生成的增强视图的距离
$$
\ell_{i,j}=-\log\frac{\exp\left({\rm sim}\left(z_{i},z_{j}\right)/{\tau}\right)}{\sum_{k=1}^{2N}1_{[k\neq i]}\exp\left({\rm sim}\left(z_{i},z_{k}\right)/{\tau}\right)}
$$

# METHODOLOGY

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230908154134370.png" alt="image-20230908154134370" style="zoom:60%;" />

## Data Augmentation

在进行数据增强时，通过在原始图数据上多次应用来生成多个视图。这种增强可以通过随机丢弃一些节点来减少高频节点对表示的影响，从而使表示可以从长尾节点学到更多

**Node Drop(ND)**
$$
V({\cal G})=({\bf M}_{1}\odot{\cal N},\cal E)
$$
这种增强随机丢弃一些边缘，预计对比学习学到的表示不会受到特定交互的影响，并且表示的鲁棒性会得到提高

**Edge Drop(ED)**
$$
V({\cal G})=({\cal N},{\bf M}_{2}\odot{\cal E})
$$
作者认为，上述两种增强方法只考虑丢弃一部分信息来缓解噪声和长尾数据的问题，而没有考虑通过添加一部分信息来增加推荐结果的多样性。为了进一步提高推荐结果的多样性，并确保每个节点的表示尽可能包含用户可能的兴趣，文章提出了一种新的数据增强方法，称为节点复制

**Node Replication(NR)**
$$
V({\cal G})=({\bf M}_{3}\odot{\cal N},{\cal E})
$$
节点复制会根据概率将当前节点的部分交互替换为相似节点对应的交互，节点之间的相似度用余弦表示，也就是说，交互历史越相似，节点就越相似
$$
S={\frac{{\mathcal{G}}{\mathcal{G}}^{T}}{|{\mathcal{G}}|}}
$$

## Supervised Comparative Learning

建立节点的多个视图后，将它们分别输入到GCN中，然后对生成的表示进行对比学习损失，与 InfoNCE 将同一节点的视图视为正样本对，并将任何其他不同节点的视图视为负样本不同，本文提出了一种监督对比学习损失，将相似的节点视为相同的类和所有这些节点都被视为正样本。我们称这种损失为监督 InfoNCE（S-InfoNCE）
$$
\ell_{i,j}=-\log\frac{\displaystyle\sum_{k=1}^{2N}1_{[j\in i]}\exp\left({\rm sim}\left({\bf z_{i}},{\bf z_{j}}\right)/\tau\right)}{\displaystyle\sum_{k=1}^{2N}1_{[k\notin i]}\exp\left({\rm sim}\left({\bf z_{i}},{\bf z_{k}}\right)/\tau\right)}
$$

## Multi-task Learning Loss of SCL

$$
{\cal L}={\cal L}_{B P R}+\lambda_{1}{\cal L}_{S-I n f o N C E}+\lambda_{2}L^{2}
$$