# Collaborative Neighbor Graph Construction

构建协作邻居图 $\hat {\mathcal G}$，通过经典的 Jaccard Similarity Coefficient
$$
s i m_{i,j}=\frac{|N_{\mathcal G}(i)\cap N_{\mathcal G}(j)|}{|N_{\mathcal G}(i)\cup N_{\mathcal G}(j)|}
$$
通过计算 user-user/item-item 之间的相似性分数，再经过一次筛选
$$
\hat{{\cal G}}_{i,j}=\left\{\begin{array}{c c}{{s i m_{i,j}}}&{{s i m_{i,j}\geq\eta \quad{\bf or}\quad top\ K\ values\ for\ node\ i}} \\ {{0}}&{{others}}\end{array}\right.
$$
$\eta$ 和 $K$ 都是超参数，每个数据集的特性导致需要设置不同的超参





# Representation Learning on Multi-Graph

## Information Propagation

使用 LightGCN 进行消息传播
$$
\begin{array}{c}{{h_{i}^{{\cal G},l}=\sum_{j\in N_{{\cal G}(i)}}\frac{1}{\sqrt{\left|N_{{\cal G}(i)}\right|\,\times\left|N_{{\cal G}(j)}\right|}}h_{j}^{l-1}}}\\ 
{{h_{i}^{{\cal G},l}=\sum_{j\in N_{{\hat{\cal G}}(i)}}\frac{1}{\left|N_{{\hat{\cal G}}(i)}\right|}h_{j}^{l-1}}}\end{array}
$$

## Adaptive High-order Information Fusion

自适应融合机制
$$
h_{i}^{l}=h_{i}^{\mathcal{G},l}+\beta_{i}\ h_{{i}}^{\hat{\mathcal G},l}
$$

$$
\beta_{i}=\frac{\gamma}{l+s i m(h_{i}^{\mathcal{G,l}},\;h_{i}^{\hat{\mathcal{G}},l})\;d_{i}}
$$

$$
d_{i}=\frac{{log}\;(|N_{{\cal G}(i)}|)}{\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}{log}\;(|N_{{\cal G}(v)}|)}
$$

设计原则

无用信息剔除

随着层数的增加，由于邻域感受野的快速扩张，高阶信息变得无用，甚至对性能产生负面影响

活动意识

对于高活跃用户，来自 G 中的一阶邻居的信息就足够

减少信息冗余



## Layer Combination and Model Prediction

平均池化过程
$$
e_{u}={\frac{1}{L+1}}\sum_{l=0}^{L}h_{u}^{l}\quad e_{i}={\frac{1}{L+1}}\sum_{l=0}^{L}h_{i}^{l}
$$







# multi-task learning

不直接将原图 $\mathcal G$ 和
$$
e_{v}^{\mathcal G}=\frac{1}{L}\sum_{l=1}^{L}h_{v}^{\mathcal G,l},\quad e_{v}^{\hat{\mathcal G}}=\frac{1}{L}\sum_{l=1}^{L}h_{v}^{\hat{\mathcal G},l}
$$

$$
\mathcal{L}_{G r a p h}^{\mathcal G}={~\sum_{{v\in\mathcal{B}}}-l o g}\frac{e x p((e_{v}^{T}e_{{v}}^{\mathcal G})/\tau)}{\sum_{j\in\mathcal{B}}e x p((e_{{v}}^{T}e_{j}^{\mathcal G})/\tau)}
$$

$$
\mathcal{L}_{G r a p h}^{\hat {\mathcal G}}={~\sum_{{v\in\mathcal{B}}}-l o g}\frac{e x p((e_{v}^{T}e_{{v}}^{\hat{\mathcal G}})/\tau)}{\sum_{j\in\mathcal{B}}e x p((e_{{v}}^{T}e_{j}^{\hat{\mathcal G}})/\tau)}
$$

$$
\mathcal{L}_{G r a p h}=\mathcal{L}_{G r a p h}^{\mathcal G} + \alpha\mathcal{L}_{G r a p h}^{\hat {\mathcal G}}
$$

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231219215804818.png" alt="image-20231219215804818" style="zoom:80%;" />

## In-Depth Analysis

没啥好说的，就是给了公式证明，反正看不懂



## Layer-level Contrastive Learning

由于添加了辅助高阶信息，AdaMCL 可能会受到噪声信号的影响

![image-20231220095043393](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231220095043393.png)
$$
{\mathcal{L}}_{L a y e r}=\sum_{v\in{\mathcal{B}}}-l o g\frac{e x p((e_{v}^{T}h_{v}^{L/2})/\tau)}{\sum_{j\in{\mathcal{B}}}e x p((e_{v}^{T}h_{j}^{L/2})/\tau)}
$$
就是把融合的信息和高阶信息进行对比，稍微过滤高阶噪声

# Optimization

总的损失函数
$$
{\mathcal{L}}={\mathcal{L}}_{B P R}+\lambda_{1}{\mathcal{L}}_{G r a p h}+\lambda_{2}{\mathcal{L}}_{L a y e r}+\lambda_{3}||\Theta||_{2}
$$
