**提出问题**

1. 稀疏交互本身意味着监督信号不足，限制了基于 GNN 的监督模型
2. 稀疏交互（CF部分）和冗余KG事实（KG部分）的结合进一步导致信息利用的不平衡
3. GNN 范式聚合局部邻居进行节点表示学习，而忽略了非局部知识图谱事实，导致知识提取不足



这里的局部就是说 1-hop? 2-hop? 所以他考虑 n 跳？

早期研究将 CF 和 KG 独立看待，采用 KGE 预训练（TransE、TransH）预训练视为先验信息

这种预训练从知识图谱中提取到的知识信号不足，因为他们独立处理 item-entity 关系

所以后面的研究侧重于通过捕获远程 KG 连接来提取更充足的 KG 信号来丰富 CF，之前有采用元路径的，但是依赖于优化手动设计的元路径

**最近的工作将 CF 和 KG 统一为异构图（Heterogeneous Graph）并使用信息聚合范式（即图神经网络，GNN）来执行图表示学习**





**进一步总结归纳问题**

1. 稀疏监督信号

   在实际场景中，用户-项目交互实际上非常稀疏，这使得它不足以达到令人满意的性能，甚至导致可怕的副作用，例如退化问题（即，将节点嵌入分布退化为窄锥体，甚至导致生成的节点表示的不区分）

2. 信息利用不平衡

   当稀疏的用户项交互遇到冗余的知识事实时，KGR中就会出现不平衡的异构结构，从而导致信息利用不平衡的问题，噪声 KG 信息在最终的用户/项目建模中得到更多强调

3. 知识提取不足

   基于 GNN 的方法通常通过在其局部 KG 结构上聚合邻近实体（即项目本身的邻近区域）来学习项目表示，这忽略了非本地 KG 事实（即相似项目的邻近区域），但是，简单地聚合非本地 KG 事实可能会引入更多不相关的噪声从而学习到更坏的表示



Straightforward idea

将输入的 user-item-entity graph 进行 DA（Data Augmentation）视为图视图（View Graph），然后将增强后的节点和原始节点进行对比。但是这种方式以相对独立的方式进行对比学习，**仅仅对比不同视图的相同部分**，**忽略了图中不同部分的内部交互**，也就是说，CF和KG表示学习的改进是相互隔离的

因此，有必要赋予对比学习范式以CF和KG部分之间有效信息交互的能力，以便在不依赖额外显式标签的情况下连贯地利用每个部分的信息



**文章提出的机制**

![image-20230816102834804](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230816102834804.png)

 (a) 图内级交互对比机制 (b) 图间级别交互对比机制

对比 CF 和 KG 部分以平衡它们对表示学习的影响，对比KG中的局部图和非局部图，提取信息丰富的非局部KG事实，KGIC 专注于为 KGR 探索一种合适的图内对比学习机制，旨在以自我监督的方式统一两个关键但相对独立的部分（即 CF 和 KG）



**KGIC 3个主要模块**

1. 图构建和编码
2. 图内交互式对比学习
3. 图间交互式对比学习



**图构建和编码**

***

局部图构建

局部图由一阶 CF（即用户交互的项目或项目本身）和用户/项目的相关 KG 事实组成。首先，从用户-项目交互中提取用户/项目的一阶 CF 信号。然后通过 item-entity 对齐 ${\mathcal{A}}=\{(v,e)|v\in {\mathcal{V}},e\in{\mathcal{E}}\} $，将一阶 CF 信号的 item 与 KG 对齐，并获取 KG 中的初始实体如下：
$$
\begin{array}{l}{{{\mathcal{E}}_{u,L}^{0}=\{e\mid(v,e)\in{\mathcal{A}},\ \mathrm{and}\ v\in\{v\mid y_{u v}=1\}\}\,,}}\\ {{{\mathcal{E}}_{v,L}^{0}=\{e\mid(v,e)\in{\mathcal{A}}\},}}\end{array}
$$
${\mathcal{E}}_{u,L}^{0}$ 表示的是用户 $u$ 与已有交互的物品对应的实体的集合。具体来说，对于集合中的每个 $e$，存在一个交互关系 $(v,e)$，其中 $v$ 是用户 $u$ 与之交互的物品，而 $e$ 是与该物品关联的实体。这个集合可以帮助建立用户已有交互的物品与实体之间的关联，以为着这个集合中的 $e$ 是特定用户交互过的 item 所对应 KG 中的 entity 集合

${\mathcal{E}}_{v,L}^{0}$ 则表示的是所有物品 $v$ 对应的实体的集合，无论是否有用户与之交互。对于集合中的每个 $e$，存在一个交互关系 $(v,e)$，表示物品 $v$ 与实体 $e$ 之间的关联。这个集合提供了所有物品与实体之间的关联信息。通过这样做，构建了 user/item 的局部图。局部图中的三元组获得如下：
$$
{\cal S}_{o,L}^{l}=\left\{\left(h,r,t\right)\mid\left(h,r,t\right)\in{\cal G}{\mathrm{~and~}}h\in{\cal E}_{o,L}^{l-1}\right\},l=1,\ldots,L,
$$
$o$ 表示统一占位符，表示 $u$ 或者 $v$

***

非局部图构建
$$
{\mathcal V}_{p}=\left\{v_{p}\mid u\in\mathcal{U}_{\mathrm{sim}}\,,\;\mathrm{and}\;y_{u v_{p}}=1\right\},
$$
寻找与一组相似用户具有共同交互历史的物品集合
$$
{\mathcal U}_{\mathrm{sim}}\,=\,\left\{u_{\mathrm{sim}}\ \bigm|\,v\,\in\,\left\{v\mid y_{u v}=1\right\}\ \,\mathrm{and}\ y_{u_{sim}v}=1\right\},
$$
表示一个集合，其中包含了一组用户 $u_{\mathrm sim}$,这些用户与特定物品 $v$ 具有共同的交互历史，所以这个集合作用是通过某个物品找到具有相似交互历史的用户
$$
{\mathcal V}_{u}=\left\{v_{u}\mid u\in\{u\mid y_{u v}=1\right\}\mathrm{~and~}y_{u v_{u}}=1 \},
$$
${\mathcal V}_p$ 和 ${\mathcal U}_{\mathrm{sim}}$ 分别表示高阶 item 和与 user 的相似用户，${\mathcal V}_u$ 表示与 user 有交互的其他物品集合


$$
{\mathcal{E}}_{u,N}^{0}=\left\{e\mid(v_{p},e)\in{\mathcal{A}},{\mathrm{~and~}}v_{p}\in {\mathcal V }_{p}\right\}
$$

$$
{\mathcal{E}}_{v,N}^{0}=\left\{e\mid(v_{u},e)\in{\mathcal{A}},{\mathrm{~and~}}v_{u}\in {\mathcal V }_{u}\right\}
$$

$$
{\cal S}_{o,N}^{l}=\left\{(h,r,t)\mid(h,r,t)\in{\cal G}\mathrm{~and
~}h\in{\cal{S}}_{o,N}^{l-1}\right\},l=1,\ldots,L.
$$

***

图编码

attentive embedding mechanism
$$
\mathbf{E}_{o,D}^{\mathit{l}}=\sum_{i=1}^{m}\pi\left(e_{i}^{\mathit{h}},r_{i}\right)e_{i}^{\!\mathit{t}}
$$
$D$ 是统一占位符，表示 $L$ (local) 或 $N$ (non-local),$\mathbf{E}_{o,D}^{\mathit{l}}$ 表示第 𝑙 层 𝑢 或 𝑣 在局部/非局部图中的嵌入
$$
\pi\left(e_{i}^{h},r_{i}\right)=\sigma\left(W_{1}\left[\sigma\left(W_{0}\left(e_{i}^{h}||r_{i}\right)+b_{0}\right)\right]+b_{1}\right),
$$

$$
\pi\left({e}_{i}^{h},{r}_{i}\right)=\frac{\exp\left(\pi\left({e}_{i}^{h},{r}_{i}\right)\right)}{\sum_{(h^{\prime},r^{\prime},t^{\prime})\in S_{o.D}^{l}}\exp\left(\pi\left({e}_{i}^{h^{\prime}},{r}_{i}^{\prime}\right)\right)},
$$



**图内交互式对比学习**

由于局部图和非局部图具有相对不平衡的异构结构，由稀疏的用户-项目交互和冗余的KG连接组成，因此关键的 CF 信号往往对表示学习的影响较小。因此，提出了图内交互式对比学习，通过对比学习执行 CF 和 KG 信息之间的交互，以连贯地使用 CF 和 KG，图内交互式对比学习将 CF 部分视为锚点，位于局部/非局部图的中心层
$$
{\mathcal L}_{intra}^{U} = \sum_{u\in {\mathcal U}}-\log\frac{\sum_{k\in L}\,e^{\left(\left(\mathbf{E}_{u,L}^{(\mathbf{0})}\cdot\mathbf{E}_{u,L}^{(k)}/\tau\right)\right)}}{\sum_{k\in L}\,e^{\left(\left(\mathbf{E}_{u,L}^{(\mathbf{0})}\cdot\mathbf{E}_{u,L}^{(k)}/\tau\right)\right)}+\sum_{k^{\prime}\in L}\,e^{\left(\left(\mathbf{E}_{u,L}^{(\mathbf{0})}\cdot\mathbf{E}_{u,L}^{(k^{\prime})}/\tau\right)\right)}}+\sum_{u\in {\mathcal U}}-\log\frac{\sum_{k\in L}\,e^{\left(\left(\mathbf{E}_{u,N}^{(\mathbf{0})}\cdot\mathbf{E}_{u,N}^{(k)}/\tau\right)\right)}}{\sum_{k\in L}\,e^{\left(\left(\mathbf{E}_{u,N}^{(\mathbf{0})}\cdot\mathbf{E}_{u,N}^{(k)}/\tau\right)\right)}+\sum_{k^{\prime}\in L}\,e^{\left(\left(\mathbf{E}_{u,N}^{(\mathbf{0})}\cdot\mathbf{E}_{u,N}^{(k^{\prime})}/\tau\right)\right)}}
$$

$$
{\mathcal{L}}_{I n t r a}={\mathcal{L}}_{I n t r a}^{U}+{\mathcal{L}}_{I n t r a}^{I}
$$
**图间交互式对比学习**

图内交互式对比学习已经在每个单个图中实现了连贯的信息利用，但由于非局部信息噪声较大，将局部和非局部信息整合在一起仍然是一个挑战,非局部图是由高阶CF信号及其对应的KG事实组成，因此包含了更多外部有用的外部事实以及噪声信息。图间交互式对比学习将局部图的任意层视为锚点，非局部图中的同一层形成正对，非局部图中的其他层是视为负对
$$
\mathcal{L}_{I n t e r}^{U}=\sum_{u\in\mathcal{U}}{{{\sum_{k\in L}^{}}}}-\log\frac{e^{\left(\left(\mathbf{E}_{u,L}^{({k})}\cdot\mathbf{E}_{u,L}^{(k)}/\tau\right)\right)}}{e^{\left(\left(\mathbf{E}_{u,L}^{({k})}\cdot\mathbf{E}_{u,L}^{(k)}/\tau\right)\right)}+\sum_{k^{\prime}\neq k}e^{\left(\left(\mathbf{E}_{u,L}^{({k})}\cdot\mathbf{E}_{u,L}^{(k^{\prime})}/\tau\right)\right)}}
$$

$$
{\mathcal{L}}_{I n t e r}={\mathcal{L}}_{I n t e r}^{U}+{\mathcal{L}}_{I n t e r}^{I}
$$

**模型预测**
$$
\hat{y}(u,i)={\bf e}_{u}^{\top}{\bf e}_{i}
$$

$$
\begin{array}{l}{{\mathbf{e}_{u}=\mathbf{E}_{u,L}^{0}||\cdot\cdot\cdot||\mathbf{E}_{u,L}^{L}||\mathbf{E}_{u,N}^{0}||\cdot\cdot\cdot||\mathbf{E}_{u,N}^{L}}}\\ {{\mathbf{e}_{i}=\mathbf{E}_{i,L}^{0}||\cdot\cdot\cdot||\mathbf{E}_{i,L}^{L}||\mathbf{E}_{i,N}^{0}||\cdot\cdot\cdot||\mathbf{E}_{i,N}^{L}}}\end{array}
$$

$$
{\mathcal{L}}_{\mathrm{BPR}}=\sum_{(u,i,j)\in O}-\ln\sigma\,\left({\hat{y}}_{u i}-{\hat{y}}_{u j}\right)
$$

$$
\mathcal{L}_{K G I C}=\mathcal{L}_{\mathrm{BPR}}+\lambda1(\alpha\mathcal{L}_{I n t r a}+\mathcal{L}_{I n t e r})+\lambda2\Vert\Theta\Vert_{2}^{2}
$$

**实验**

基本套路

按照 RippleNet 将三个数据集中的显式反馈转换为隐式反馈，其中 1 表示正样本。对于负样本，为每个用户随机抽取与正样本大小相同的未观察到的项目

知识图谱的构建，使用 Microsoft Satori 4 ，紧随 RippleNet 和 KGCN 。每个子KG都遵循三重格式，并且是整个KG的子集，置信度超过0.9。给定子 KG，通过将名字与三元组的尾部进行匹配来收集所有有效电影/书籍/音乐家的 Satori ID。然后我们将项目 ID 与所有三元组的头部进行匹配，并从子 KG 中选择所有匹配良好的三元组

评估指标

在点击率（CTR）预测中，采用了两个广泛使用的指标 𝐴𝑈𝐶 和 𝐹1

在 Top-K 预测中，选择 Recall@𝐾 来评估推荐的集合，其中 𝐾 设置为 5、10、20、50 和 100 以保持一致性

参数设置

所有模型中的嵌入大小固定为 64。采用默认的 Xavier 方法来初始化模型参数



***

RQ1：与最先进的模型相比，KGIC 的表现如何

1. 通过将 CF 与局部/非局部图中的 KG 信号进行对比，图内级别的交互式对比学习在两个部分之间进行交互并相互监督以改进表示学习
2. 通过对比用户/项目的局部图和非局部图，图间级别的交互式对比学习充分结合了非局部知识图谱事实，并从两种图中学习判别表示
3. 在大多数情况下，提取更多信息量的知识图谱事实可以提高模型性能，基于 GNN 的方法比基于嵌入和基于路径的方法具有更好的性能，这表明提取远程知识图谱事实的有效性

***

RQ2：主要组成部分（例如图内和图间交互式对比学习）如何影响 KGIC 性能？

![image-20230820163448530](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230820163448530.png)

***

RQ3：不同的超参数设置（例如模型深度、系数𝛼、温度𝜏）如何影响KGIC？

模型深度的影响

![image-20230820170912317](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230820170912317.png)

1. 一层或两层是聚合局部/非局部图中的邻近信息的适当距离，进一步堆叠更多层只会引入更多噪声
2. 一层或两层的正对足以在 CF 和 KG 层之间进行交互时学习判别嵌入



系数 $\alpha$ 的影响

![image-20230820171646507](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230820171646507.png)

看出一般就取值为 1 的时候在各个数据集上表现最好，此外，在不同的 𝛼 下，KGIC 的性能始终优于其他基线，这也证实了多级交互式对比学习机制的有效性。



温度系数 $\tau$ 的影响

温度 𝜏 的影响。在 {0.05, 0.075, 0.1, 0.2, 0.3, 0.4} 的范围内变化。所示的结果中，发现：太大的 𝜏 值会导致性能较差，与之前的工作的结论一致。一般来说，[0.1,0.2]范围内的温度可以获得令人满意的推荐性能

***

RQ4：自监督任务真的能提高表征学习质量吗？

为了更直观地评估所提出的交互式对比机制如何影响表示学习性能，可视化学习项目嵌入，之前的对比学习工作使用高斯核密度估计（KDE）在二维空间中绘制项目嵌入分布（其中颜色越深，落在该区域的点越多），并在角度上绘制了 KDE（即中每个点 (𝑥, 𝑦) 的 𝑎𝑟𝑐𝑡𝑎𝑛2(𝑦, 𝑥)），以便更清晰地呈现

![image-20230820195031357](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230820195031357.png)

E:\PaperCode\KACL\KACL-dgl\Model\main.py

E:\PaperCode\KACL\KACL-dgl\Model

/home/tsx/paperCode/KACL/KACL-dgl/Model
