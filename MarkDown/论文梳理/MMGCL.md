#  ABSTRACT

提出了一种名为多模态图对比学习（MMGCL）的新学习方法，旨在以自监督学习的方式显式增强多模态表示学习

# METHODOLOGY

## Problem Definition and Notations

### Problem Definition and Notations

$$
m\,\in\,{\cal M}\,=\,\{v,a,t\}
$$

v, a, t 被定义为三个模态，分别是 visual, acoustic, textual，根据这三个不同的模态分别生成多模态交互图 ${\mathcal G}_v,{\mathcal G}_a,{\mathcal G}_t$

### Multi-view Graph Augmentation

受之前工作的启发，本文在多模态图上设计了两个增强算子 modality edge dropout 和 modality masking

**Modality Masking**：对 user/item 特征的特定模态应用屏蔽模式，随机概率丢弃某个模态的信息
$$
V_{1}(\mathcal{G})=\left\{\begin{array}{l l}{{(\mathcal{V}_{v},\mathcal{E}_{v})\parallel(\mathcal{V}_{a},\mathcal{E}_{a})\parallel(M_{1}\odot\mathcal{V}_{t},\mathcal{E}_{t})}}&{{\mathrm{with~}p_{t}}}\\ {{(\mathcal{V}_{v},\mathcal{E}_{v})\parallel(M_{1}\odot\mathcal{V}_{a},\mathcal{E}_{a})\parallel(\mathcal{V}_{t},\mathcal{E}_{t})}}&{{\mathrm{with~}p_{a}}}\\ {{(M_{1}\odot\mathcal{V}_{v},\mathcal{E}_{v})\parallel(\mathcal{V}_{a},\mathcal{E}_{a})\parallel(\mathcal{V}_{t},\mathcal{E}_{t})}}&{{\mathrm{with~}p_{v}}}\end{array}\right.
$$
通过在输入层中用随机初始化的嵌入替换 user/item 特征的特定模态来实现此掩码运算符

**Modality Edge Dropout**：以丢失率 𝜌 随机删除每个模态图中的边缘

## Challenging Negative Samples

给定样本集合 $\left\{s_{1}^{i},s_{2}^{i}\right\}_{i=1}^{\mathcal N} $ ,构建正样本对 $x\,=\,\left\{s_{1}^{i},s_{2}^{i}\right\} $，负样本对 $y\,=\,\left\{s_{1}^{i},s_{2}^{j}\right\} $

举个例子，给定一个锚点样本 $s^1_1$，包含三个模态 $\left(c_{1,i}^{v},c_{1,i}^{a},c_{1,i}^{t}\right)$，正样本 $s^1_2$，$\left(c_{2,i}^{v},c_{2,i}^{a},c_{2,i}^{t}\right) $，扰动负样本 $s^j_2$ $\left(c_{2,j}^{v},c_{2,d(j)}^{a},c_{2,j}^{t}\right) $

## Contrastive Learning

$$
\mathcal{L}_{s s l}^{u s e r}=-{\mathbb E}_{\{s_{1}^{1},s_{2}^{1},...,s_{2}^{k+1}\}}\left[\log\frac{h(\{s_{1}^{1},s_{2}^{1}\})}{\sum_{j=1}^{k+1}(h(\left\{s_{1}^{1},s_{2}^{j}\right\})}\right]
$$

$$
h(\{s_{1}^{1},s_{2}^{1}\})=\exp{(\frac{f(V_{1}(\mathcal{G}))\cdot f(V_{2}(\mathcal{G}))}{\|f(V_{1}(\mathcal{G}))\|\cdot\|f(V_{2}(\mathcal{G}))\|}\cdot\frac{1}{\tau})}
$$

![image-20230911204225313](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230911204225313.png)

所以这篇文章所提出的模型就是在原始图的基础上，首先用 Masking 方法生成 Anchor，然后分别利用 Edge Dropout 和 Modal Perturb 生成正负样本对，之后利用 Multimodal Encoder 编码生成嵌入表示，再将其输入到 InfoNCE 中进行对比损失训练，这里的 Encoder 没写，估计就是类 GCN 方法