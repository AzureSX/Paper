# Are Graph Augmentations Necessary

**为什么要在推荐系统中使用对比学习**

对比学习（CL）最近在推荐领域推动了一系列卓有成效的研究，因为它从原始数据中提取自我监督信号的能力非常符合推荐系统解决**数据稀疏**问题的需求

**基于对比学习的推荐系统的典型流程**

基于 CL 的推荐模型的典型流程是首先使用结构扰动来增强用户-项目二分图，然后最大化不同图增强之间的节点表示一致性。尽管这种范例被证明是有效的，但性能提升的背后是什么仍然是一个谜，也就是说仍然是一个黑箱

**创新点**

提出了一种简单的 CL 方法，该方法放弃图增强，而是向嵌入空间添加均匀噪声以创建对比视图

**一般对比学习范式**

![image-20230802095533380](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230802095533380.png)

**研究出发点**

一些最新的工作报道说，即使是 CL 中极其稀疏的图增强（边缘丢失率为 0.9）也可以带来期望的性能增益。这种现象非常难以捉摸且违反直觉，因为较大的丢失率将导致原始信息的巨大损失和高度倾斜的图结构，也就是说基于 dropout 的图增强实际上的作用微乎其微？经过研究，文章认为对推荐性能真正重要的是 CL 损失，而不是图增强

**技术分析**

遵循图 1 所示的图 CL 框架，但放弃了基于 dropout 的图增强，而是向原始表示添加随机均匀噪声，以实现表示级数据增强



**InfoNCE Loss**

对比学习损失函数有多种，其中比较常用的一种是InfoNCE loss，InfoNCE loss其实跟交叉熵损失有着千丝万缕的关系，下面借用恺明在论文 MoCo 里定义的InfoNCE loss公式来说明。论文 MoCo 提出，我们可以把对比学习看成是一个字典查询的任务，即训练一个编码器从而去做字典查询的任务。假设已经有一个编码好的 query $q$（一个特征），以及一系列编码好的样本 $k0,k1,k2$...，那么 $k0,k1,k2$...可以看作是字典里的 key。假设字典里只有一个key即 $k+$(称为 $k$ positive）是跟 $q$ 是匹配的，那么 $q$ 和 $k+$ 就互为正样本对，其余的 key 为 $q$ 的负样本。一旦定义好了正负样本对，就需要一个对比学习的损失函数来指导模型来进行学习。这个损失函数需要满足这些要求，即当query $q$ 和唯一的正样本 $k+$ 相似，并且和其他所有负样本 key 都不相似的时候，这个loss的值应该比较低。反之，如果 $q$ 和 $k+$ 不相似，或者 $q$ 和其他负样本的 key 相似了，那么loss就应该大，从而惩罚模型，促使模型进行参数更新
$$
L_{q}=-l o g\frac{e x p(q\cdot k_{+}/\tau)}{\sum_{i=0}^{k}e x p(q\cdot k_{i}/\tau)}
$$
其中的 $q$ 和 $k$ 可以表示为其他的形式，比如相似度度量，余弦相似度等。分子部分表示正例之间的相似度，分母表示正例与负例之间的相似度，因此，相同类别相似度越大，不同类别相似度越小，损失就会越小
$$
e x p(q\cdot k_{+}/\tau)
$$
温度系数 $\tau$ 是设定的超参数，它的作用是控制模型对负样本的区分度。温度系数设的越大，q*k的分布变得越平滑，那么对比损失会对所有的负样本一视同仁，导致模型学习没有轻重。如果温度系数设的过小，则模型会越关注特别 hard 负样本，但其实那些负样本很可能是潜在的正样本，这样会导致模型很难收敛或者泛化能力差。因此温度系数的设定是不可或缺的

**那么 Graph Augmentations 的价值和意义呢** 

尽管没有预期的那么有效，但从原始图的适当扰动版本有助于学习对扰动因子不变的表示这一意义上说，图扩充并非完全无用。这就类似 KGCL 论文里提到的“一致性”，可以筛选出对扰动不那么敏感的 user/item

**由 SGL 给出的一般基于对比学习的联合损失**
$$
{\cal{L}}_{j o i n t}={\cal{L}}_{r e c}+\lambda{\cal{L}}_{c l}
$$
$\cal{L}_{cl}$ 就是上面给出的 InfoNEC，具体来说就是以下公式
$$
{\mathcal{L}}_{c l}=\sum_{i\in{\mathcal{B}}}-\log{\frac{\exp({\mathbf{z}}_{i}^{\prime\top}{\mathbf{z}}_{i}^{\prime\prime}/\tau)}{\sum_{j\in{\mathcal{B}}}\exp({\mathbf{z}}_{i}^{\prime\top}{\mathbf{z}}_{j}^{\prime\prime}/\tau)}}
$$
$i, j$ 是属于同一 $\mathcal{B}$(batch），${\mathbf{z}}_{i}^{\prime}{\mathbf{z}}_{i}^{\prime\prime}$ 是从两种不同的基于 dropout 的图增强学习到的 𝐿2 归一化 𝑑 维节点表示，$\tau$ 是温度常数
$$
{\bf E}=\frac{1}{1+L}\left({\bf E}^{(0)}+\tilde{\bf A}{\bf E}^{(0)}+...+\tilde{\bf A}^{L}{\bf E}^{(0)}\right)
$$
之后再通过 LightGCN 实现消息传播聚合，$\frac{1}{1+L}$ 是 LightGCN原文中尝试出的参数，所以不需要针对每一层特殊设计
$$
{\bf z}_{i}^{\prime}=\frac{\mathbf{e}_{i}^{\prime}}{||\mathbf{e}_{i}^{\prime}||_{2}}
$$
$\mathbf{e}_{i}^{\prime}$ 是 $\bf E$ 中 $\bf e$ 的 corrupted 版本，$
{{\bf E}^{(0)}}\ \ \in\ \ {{\mathbb{R}^{|N|\times d }}}\
$

**为什么不对原始图进行任何数据增强的操作也能获得比较好的效果**

作者推测，不一定为最终结论，节点丢失和随机游走（尤其是前者）很可能会丢失关键节点和相关边，从而将相关子图分解成不相连的部分，这会严重扭曲原始图，即节点丢失相对于边丢失会有较大的风险

**文章的出发点**

文献 [25] 指出优化对比损失强化了视觉表示学习中的两个属性：正对特征的对齐，以及单位超球面上归一化特征分布的均匀性（听不懂）。关于这两个特性对于推荐是不是也有作用，但是作者还是利用 [25] 中的“可视化”方法来研究“均匀性”

**文章研究“均匀性”的方法**

先用 t-SNE 技术将数据可视化，t-SNE（t-distributed Stochastic Neighbor Embedding）是一种非线性降维和数据可视化的技术，用于将高维数据映射到低维空间，以便于可视化和观察数据之间的关系。t-SNE 的降维结果通常用于数据可视化，特别是对于高维数据的可视化。通过将数据映射到二维或三维空间，我们可以在更直观的视角下观察数据之间的关系、聚类结构和类别分布等信息。t-SNE 在许多领域，特别是在机器学习和数据挖掘中，都被广泛应用于数据分析和可视化。

t-SNE 的工作原理如下：

1. 首先，t-SNE 使用高斯分布（Gaussian distribution）来表示高维空间中的样本之间的相似性，这种相似性通过计算样本之间的概率来表示。
2. 然后，t-SNE 在低维空间中同样使用高斯分布来表示样本之间的相似性。在低维空间中，通过随机初始化样本的位置，t-SNE 开始优化样本的位置，使得低维空间中样本之间的相似性与高维空间中的相似性尽可能匹配。
3. 优化过程通过最小化 Kullback-Leibler 散度（KL 散度）来完成。KL 散度是一种衡量两个概率分布之间差异的指标。t-SNE 使用 KL 散度来度量高维空间和低维空间中样本之间的相似性差异，并不断调整低维空间中样本的位置，使得 KL 散度最小化。

换句话说就是将各种方法达到最优时的获得的所有 representations 二维化可视化（同时还将单位圆上的“角度密度”估计也可视化，不懂）

![image-20230804190401686](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230804190401686.png)

**造成高度聚集的特征分布的原因**

作者认为有两个原因，首先是LightGCN中的消息传递机制。随着层数的增加，节点嵌入变得“局部相似”，其二是推荐数据中的流行度偏差[4]，通过计算梯度，由于推荐数据通常遵循长尾分布，当 𝑖 是具有大量交互的热门项目时，用户嵌入将不断向 𝑖 的方向更新（即-∇e𝑢）

**为什么均匀分布会对推荐效果有提升**

优化 CL 损失可以被视为一种隐式的去偏方法，因为更均匀的表示分布可以保留节点的内在特征并提高泛化能力

**SGL影响因素**

分布的均匀性是SGL中对推荐性能产生决定性影响的根本因素

**SimGCL**

基于之前的发现，作者推测通过在一定范围内调整学习表示的均匀性，可以达到最佳性能。作者提出的SimGCL可以平滑地调节均匀性并提供信息方差，以最大限度地提高 CL 的收益

**SimGCL的动机**

由于操纵图结构以获得更均匀分布的表示空间既困难又耗时（图数据增强操作），因此将注意力转移到嵌入空间。受对抗性示例 [8] 的启发，对抗性示例 [8] 是通过向输入图像添加难以察觉的小扰动而构建的，直接向表示添加随机噪声，以实现高效且有效的增强。

[8] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014. Explaining and harnessing adversarial examples (2014). arXiv preprint arXiv:1412.6572 (2014).

**SimGCL核心思想**

通过实验揭示，在基于CL的推荐模型中，CL通过学习更均匀分布的 user/item 表示可以隐式减轻流行度偏差（长尾？）

**SimGCL是怎么实现的**

基于 [8] 的思想，给定节点 $i$ 以及其在 $d$ 维嵌入空间中的表示 
$$
\mathbf{e}_{i}^{\prime}=\mathbf{e}_{i}+{{\Delta}}_{i}^{\prime},~~\mathbf{e}_{i}^{\prime\prime}=\mathbf{e}_{i}+{{\Delta}}_{i}^{\prime\prime}
$$
$||\Delta||_{2}=\epsilon$,控制大小，限制了噪声的数值大小是半径为 $\epsilon$ 的超球面上的点

$\Delta\,=\,\bar{\Delta}\;{\odot}\;\mathrm{sign}({\mathbf e}_{i}),\;\bar{\Delta}\;\in\;{\mathbb R}^{d}\;\sim\;U(0,1) $,

向量 ${\Delta}$ 的每个元素是由一个在区间[0,1]上均匀分布的随机数 $\bar{\Delta}$ 乘以向量 ${\mathbf e}_i$ 的对应元素的符号得到的。换句话说，$\Delta$ 的每个元素取决于对应位置上随机数的正负

**调节均匀度**

[25]中，提出了一个度量来衡量表示的均匀性，它是平均成对高斯势的对数（又名径向基函数（RBF）内核），作者选择流行的项目（超过200次交互），并在数据集中随机抽取5000个用户形成 user-item 对，然后用 [25] 提出的公式计算它们在不同模型之间的均匀性，实验结果表明在达到最优性能时，SimGCL的分布最均匀