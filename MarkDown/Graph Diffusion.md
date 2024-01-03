# IID And OOD

独立同分布，以图像数据集举例

**独立性（Independence）：** 在图像数据集中，独立性通常表示图像样本之间的内容和特征是相互独立的。例如，在一个图像分类任务中，每张图片代表一个样本，独立性假设表示一张图片的内容和特征不受其他图片的影响。这使得模型能够学到通用的特征，而不是针对数据集中特定图片之间的关系

**同分布性（Identically Distributed）：** 同分布性意味着图像样本来自相同的概率分布。在图像分类任务中，这意味着训练集和测试集中的图像应该来自相似的场景和视觉特性，以确保模型在训练时学到的特征能够在测试时有效地应用

非独立同分布，以 Graph 数据集为例

**结构依赖性：** 在图中，节点和边的连接关系构成了图的拓扑结构。节点之间的连接性以及边的权重都可能由于特定的任务或应用而发生变化。这使得节点之间的特征不再独立，而是受到图的拓扑结构的影响

**局部性：** 图中的节点通常与其相邻节点之间有较强的关联。这种局部性意味着一个节点的特征可能依赖于其相邻节点的特征，导致样本之间的依赖性

**异构性：** 图中的节点和边可以表示多种类型的实体和关系，形成异构图。这使得不同类型的节点和边之间的连接性和特征分布可能不同，破坏了同分布性的假设



# Denoising Diffusion Model

扩散生成范式可以建模为用变分推理训练的马尔科夫链（markov chains trained with variational inference）

由两个主要阶段组成，即前向扩散和反向扩散。主要思想是先构建一个噪声模型，通过添加噪声（即通常是高斯噪声）来扰乱原始输入数据，然后训练可学习的逆过程以从噪声中恢复原始输入数据



# Deep Generative Models on Graphs

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203150721511.png" alt="image-20231203150721511" style="zoom:80%;" />



## 符号表示

在图表示学习中，一张图 $\mathbf{G}=\left(\mathbf{X},\mathbf{A}\right)$ 由邻接矩阵 $\mathbf{A}\in \mathbb{R}^{N\times N}$ 和节点特征 $\mathbf{X}\in \mathbb{R}^{N\times d}$

在扩散的条件下，$\mathbf{G}_{0}$ 指原始输入图，而 $\mathbf{G}_{t}$ 指 $t$ 时间步的噪声图



## VAEs

变分自编码器旨在训练一个概率图编码器 $q_{\phi}(\mathbf{z}|\mathbf{G})$ 将图空间映射到低维连续嵌入 $\mathbf{z}$ 和图解码器 $p_{\phi}(\mathbf{G}|\mathbf{z})$ 在给定 $\mathbf{z}$ 采样的情况下重建新数据

## GAN

生成对抗网络使用两个深度神经网络：生成器 $f_{G}$ 和鉴别器 $f_{D}$，生成器尝试学习图分布并生成新图，而鉴别器尝试区分真实图和生成图



尽管取得了巨大的成功，大多数现有的深度生成模型仍然面临着图生成的挑战。例如，VAE 模型基于可能性生成图，这需要大量的图匹配过程或在实现排列不变性时对每个可能的节点对齐的可能性进行显式估计。在实践中，基于 GAN 的图生成模型很容易陷入模式崩溃，这会限制生成图的规模和新颖性



## Score Matching with Langevin Dynamics (SMLD)

## Score-based Generative Model (SGM)

## Denoising Diffusion Probabilistic Model (DDPM)

**一般来说是概率扩散去噪模型**

### 前向过程





<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203194001660.png" alt="image-20231203194001660" style="zoom:50%;" />

$x$：随机变量

$f(x)$：概率密度函数

$\pi/e$：圆周率/自然常数

$\mu$：随机变量数学期望，平均值
$$
\mu={\frac{\sum_{i=1}^{N}x_{i}}{N}}
$$
$\sigma$：随机变量的标准差，反映离散程度
$$
\sigma=\sqrt{\frac{\sum_{i=1}^{N}(x_{i}-\mu)^{2}}{N}}
$$
${\sigma}^{2}$：方差，标准差的平方
$$
\sigma=\frac{\sum_{i=1}^{N}(x_{i}-\mu)^{2}}{N}
$$
高斯分布：正态分布 $N(\mu,{\sigma}^{2})$ 的概率密度函数

高斯噪声：基于高斯分布生成的随机变量

扩散现象：指物质粒子从高浓度区域向低浓度区域移动的过程

扩散模型：通过向图片中加入高斯噪声来模拟扩散现象，并通过逆向过程生成图片

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203200244416.png" alt="image-20231203200244416" style="zoom:80%;" />

前向加噪：通过归一化操作将三通道的值映射到 [-1,+1] 区间

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203200636936.png" alt="image-20231203200636936" style="zoom:80%;" />

基于上述提到的高斯分布随机生成对映图片大小的噪声

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203202037288.png" alt="image-20231203202037288" style="zoom:50%;" />

然后将生成的噪声和原图的对应数值进行组合，$\beta$ 是一个介于 [0, 1] 之间的数字，用于产生 $\epsilon$ 和 $x$ 的系数

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203202354835.png" alt="image-20231203202354835" style="zoom:50%;" />

$\beta$ 可以看作直径为 1 的半圆的内接直角三角形的两条直角边

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203203819171.png" alt="image-20231203203819171" style="zoom:50%;" />

扩散模型可迭代，如同之前描述的扩散过程，每一小步的 $\epsilon$ 都是基于标准正态分布重新采样的随机数
$$
x_{t}=\sqrt{\beta_{t}}\times\epsilon_{t}+\sqrt{1-\beta_{t}}\times x_{t-1}
$$
且每一步中的 $\beta_{t}$ 从一个接近 0 的数字逐渐递增 
$$
0<\beta_{1}<\beta_{2}<\beta_{3}<\beta_{t-1}<\beta_{t}<1
$$
<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203204343254.png" alt="image-20231203204343254" style="zoom:50%;" />

用 $\alpha_{t} = 1 - \beta_{t}$ 替换
$$
x_{t}=\sqrt{1-\alpha_{t}}\times\epsilon_{t}+\sqrt{\alpha_{t}}\times x_{t-1}
$$
分析 $x_{t}$ 和 $x_{t-1}$ 之间的关系
$$
x_{t}=\sqrt{1-\alpha_{t}}\times\epsilon_{t}+\sqrt{\alpha_{t}}\times x_{t-1}\\
x_{t-1}=\sqrt{1-\alpha_{t-1}}\times\epsilon_{t-1}+\sqrt{\alpha_{t-1}}\times x_{t-2}\\

x_t=\sqrt{\alpha_t(1-\alpha_{t-1})}\times\epsilon_{t-1}+\sqrt{1-\alpha_{t}}\times \epsilon_{t}+\sqrt{\alpha_{t}\alpha_{t-1}}\times x_{t-2}
$$
注意 $x_{t}$ 中的 $\epsilon_{t}$ 和 $\epsilon_{t-1}$ 表示两个独立的随机变量，这里有个知识，即(正态分布+正态分布=正态分布/均匀分布+均匀分布=正态分布)，所以如果知道叠加后的概率分布，那只需要采样一次即可

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231203214357219.png" alt="image-20231203214357219" style="zoom:50%;" />

对两个概率分布进行卷积操作等于计算这两个分布的所有可能组合的情况，等效于得到叠加后的概率分布
$$
N(\mu_{1},\sigma_{1}{}^{2})+N(\mu_{2},\sigma_{2}{}^{2})=N(\mu_{1}+\mu_{2},\sigma_{1}{}^{2}+\sigma_{2}{}^{2})
$$
由于之前提到噪声是根据标准正态分布采样得到的
$$
x_t=\sqrt{\alpha_t(1-\alpha_{t-1})}\times\epsilon_{t-1}+\sqrt{1-\alpha_{t}}\times \epsilon_{t}+\sqrt{\alpha_{t}\alpha_{t-1}}\times x_{t-2} \\
N(0, \alpha_t-\alpha_t\alpha_{t-1}) + N(0, 1-\alpha_t)=N(0,1-\alpha_t\alpha_{t-1})
$$
所以可以将原来的式子改写为以下公式,这种方式又称为重参数技巧
$$
x_t=\sqrt{1-\alpha_t\alpha_{t-1}}\times\epsilon+\sqrt{\alpha_{t}\alpha_{t-1}}\times x_{t-2}
$$
目标是直接得到从 $x_0$ 到 $x_t$ 的关系式
$$
x_t = \sqrt{1-\alpha_{t}\alpha_{t-1}\alpha_{t-2}\alpha_{t-3}...\alpha_{2}\alpha_{1}}\times\epsilon+\sqrt{\alpha_{t}\alpha_{t-1}\alpha_{t-2}\alpha_{t-3}...\alpha_{2}\alpha_{1}}\times x_{0}
$$
用 $\bar{\alpha}_{t}$ 替代这一串连乘 
$$
\bar{\alpha}_{t} = \alpha_{t}\alpha_{t-1}\alpha_{t-2}\alpha_{t-3}...\alpha_{2}\alpha_{1}\\
x_t = \sqrt{1-\bar{\alpha}_{t}}\times\epsilon+\sqrt{\bar{\alpha}_{t}}\times x_{0}
$$

### 反向过程

之前定义了后一时刻与前一时刻的关系，然后根据此关系得出了原始数据 $x_0$ 在加噪后到任意时刻 $x_t$ 数据的关系，反向过程的目标是从最后 $x_t$ 时刻的噪声图片中恢复得到 $x_0$ 时刻的原图

贝叶斯公式

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231204201607068.png" alt="image-20231204201607068" style="zoom:50%;" />
$$
P(A|B)={\frac{P(B|A)P(A)}{P(B)}}
$$
$P(A)$：随机事件 $A$ 发生的概率

$P(B)：$随机事件 $B$ 发生的概率

$P(B|A)$：$A$ 事件发生的情况下 $B$ 事件发生的概率

$P(A|B)$：$B$ 事件发生的情况下 $A$ 事件发生的概率

在上述例子中，$P(A)$ 表示为小明坐公交或者地铁的概率，基于之前的经验，所以称之为“先验概率”(Prior)，$P(A|B)$ 同样表示小明坐公交或者地铁的概率，但是是在 $B$ 事件发生后对先验概率 $P(A)$ 的修正，所以称之为“后验概率”(Posterior)，这种修正的基础是看到了 $B$ 事件的发生，所以 $B$ 事件称之为“证据”(Evidence)，$P(B|A)$ 表示在 $A$ 事件发生的前提下，$B$ 事件似乎很有可能发生，所以称之为“似然”(Likelihood)，可以看作 $B$ 事件对 $A$ 事件的归因力度，当 $P(B|A)$ 的值越大，$B$ 事件就提供更强的证据支持 $A$ 事件

因为从 $x_{t-1}$ 到 $x_{t}$ 时刻是一个随机过程，所以从 $x_t$ 到 $x_{t-1}$ 也是一个随机过程，目标是已知 $x_t$ 时刻的数据求出 $x_{t-1}$ 时刻的数据

![image-20231204204141086](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231204204141086.png)

所以可以把问题抽象为
$$
P(x_{t-1}|x_{t})={\frac{P(x_{t}|x_{t-1})P(x_{t-1})}{P(x_{t})}}
$$
同时为了严谨，将 $x_0$ 这个前提条件补全
$$
P(x_{t-1}|x_{t},x_{0})={\frac{P(x_{t}|x_{t-1},x_{0})P(x_{t-1}|x_{0})}{P(x_{t}|x_{0})}}
$$
给定 $x_{t-1}$ 时刻，求 $x_t$ 
$$
{P(x_{t}|x_{t-1},x_{0})}\\
x_{t}=\sqrt{1-\alpha_{t}}\times\epsilon_{t}+\sqrt{\alpha_{t}}\times x_{t-1}\\
N({\sqrt{\alpha_{t}}} x_{t-1},1-\alpha_{t})
$$
给定 $x_{0}$ 时刻，求 $x_t$ 
$$
{P(x_{t}|x_{0})}\\
x_t = \sqrt{1-\bar{\alpha}_{t}}\times\epsilon+\sqrt{\bar{\alpha}_{t}}\times x_{0}\\
N({\sqrt{\bar{\alpha}_{t}}x_{0}},1-\bar{\alpha}_{t})
$$
同理
$$
{P(x_{t-1}|x_{0})}\\
x_{t-1} = \sqrt{1-\bar{\alpha}_{t-1}}\times\epsilon+\sqrt{\bar{\alpha}_{t-1}}\times x_{0}\\
N({\sqrt{\bar{\alpha}_{t-1}}x_{0}},1-\bar{\alpha}_{t-1})
$$
将以上正态分布函数展开
$$
N(\mu,\sigma^{2}),f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}
$$

$$
P(x_{t-1}|x_{t},x_{0})\sim{N}\left(\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})}{1-\bar{\alpha}_{t}}x_{0},\left(\frac{\sqrt{1-\alpha_{t}}\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{1-\bar{\alpha}_{t}}}\right)^{2}\right)
$$

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231204220612216.png" alt="image-20231204220612216" style="zoom:50%;" />

目标是从 $x_{T}$ 迭代直到 $x_{0}$，但是 $x_0$ 出现在以上公式中
$$
x_t = \sqrt{1-\bar{\alpha}_{t}}\times\epsilon+\sqrt{\bar{\alpha}_{t}}\times x_{0}\\
x_{0}=\frac{x_{t}-\sqrt{1-{\bar{\alpha}}_{t}}\times\epsilon}{\sqrt{\bar{\alpha}_{t}}}\\
P(x_{t-1}|x_{t},x_{0})\sim{N}\left(\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})}{1-\bar{\alpha}_{t}}\times \frac{x_{t}-\sqrt{1-{\bar{\alpha}}_{t}}\times\epsilon}{\sqrt{\bar{\alpha}_{t}}},\left(\frac{\sqrt{1-\alpha_{t}}\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{1-\bar{\alpha}_{t}}}\right)^{2}\right)
$$
对于任意的 $x_t$ 时刻的数据，都可以认为是某个 $x_0$ 原始数据直接加噪得来，从以上公式可以得出只要知晓 $\epsilon$ ，就能得到前一时刻 $x_{t-1}$ 的概率分布，这里可以训练一个神经网络模型，来预测这个噪声 $\epsilon$

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205094949255.png" alt="image-20231205094949255" style="zoom:50%;" />

那么如何得到 $x_{T}$ 的数据呢，因为 $\bar{\alpha}_{t}$ 是逐渐递减的，所以可以将 $x_{T}$ 的数据近似为标准正态分布，所以只要用标准正态分布随机采样就能生成 $x_{T}$ 时刻的图片
$$
x_t = \sqrt{1-\bar{\alpha}_{t}}\times\epsilon+\sqrt{\bar{\alpha}_{t}}\times x_{0}\\
\bar{\alpha}_{T}\approx0,x_{T}\approx\epsilon
$$


## 另一视角

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205104535788.png" alt="image-20231205104535788" style="zoom:50%;" />

除了输入图片数据之外，还要输入一个数字代表噪声程度的大小，将这两部分同时输入到 Noise Predicter 中得到预测的 Noise 然后两部分相减得到 Denoise 的输出

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205141636050.png" alt="image-20231205141636050" style="zoom:50%;" />

如何训练 Noise Predicter

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205142043439.png" alt="image-20231205142043439" style="zoom:50%;" />

这就是之前提到的前向过程

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205143239901.png" alt="image-20231205143239901" style="zoom:50%;" />

整体过程

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205153741769.png" alt="image-20231205153741769" style="zoom:50%;" />

直观上VS实际上

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205153934349.png" alt="image-20231205153934349" style="zoom:50%;" />

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231205154906078.png" alt="image-20231205154906078" style="zoom:50%;" />

### 数学原理

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231206203319154.png" alt="image-20231206203319154" style="zoom:50%;" />

**Algorithm 1**

1：重复

2：从数据集中采样一个原始数据 $\mathbf{x}_0$(clean image)

3：从 1 到 T 之间采样一个时间步 $t$(数字)

4：从正态分布中采样一个噪声 $\epsilon$

5：将 $x_{0},\epsilon,\bar{\alpha}_{t},t$ 输入到参数为 $\theta$ 的 $\epsilon_{\theta}$ (Noise Predictor)中

**Algorithm 2**

1：从正态分布中采样一个样本 $x_{T}$

2：进行 $T$ 次循环

3：再次从正态分布中采样一个 $\mathbf{z}$

4：根据公式得 $\mathbf{x}_{t-1}$

## KL散度&交叉熵

$$
\begin{array}{c}{{D_{K L}(P||Q)=\displaystyle\sum_{x}p(x)\log\frac{p(x)}{q(x)}}}\\ {{=\displaystyle\sum_{x}p(x)(log p(x)-\log q(x))}}\\ {{\displaystyle=\displaystyle\sum_{x}(p(x)\log p(x)-p(x)\log q(x))}}\end{array}\\
$$

$$
C r o s s E n t r o p y(P,Q)=-\sum_{x}p(x)\log{q(x)}=-\mathbb{E}_{p(x)}\log{q(x)} 
$$

# DDPM on Graphs

去噪扩散概率模型对图的适应主要集中在设计适当的马尔可夫链转移核，去噪扩散概率模型对图的适应主要集中在设计适当的马尔可夫链转移核，在每个扩散步骤中，图的邻接矩阵的每一行都以单热方式编码，并与双随机矩阵 $\mathbf{Q}_{t}$ 相乘，在反向过程中，模型包含重新加权的 ELBO 作为损失函数以获得稳定的训练，噪声图的条件概率如下表示
$$
q(\mathbf{G}_{t}|\mathbf{G}_{t-1})=(\mathbf{X}_{t-1}\mathbf{Q_{t}^{X}},\mathbf{E}_{t-1}\mathbf{Q}_{t}^{\mathbf E}), 
$$

$$
q(\mathbf{G}_{t}|\mathbf{G})=(\mathbf{X}\bar{\mathbf{Q}}_{t}^{\mathbf{X}},\mathbf{E}\bar{\mathbf{Q}}_{t}^{\mathbf{E}}),
$$





# CF Diffusion

**ill-posed inverse problem**

不适定问题的概念:一个数学物理定解问题的解存在、唯一并且稳定的则称该问题是适定的（Well Posed）.如果不满足适定性概念中的上述判据中的一条或几条，称该问题是不适定的



扩散模型在解决不适定逆问题方面的成功，引入了用于协同过滤的条件扩散框架，该框架在历史交互的指导下迭代地重建用户的隐藏偏好



具有隐式反馈的协同过滤是推荐系统中的一项基本技术，典型的解决方案包括图神经网络（GNN）和 自编码器 (AE),其中一些模型在优化过程中使用成对排名损失作为用户偏好得分的代理，其他方法则通过最小化重建交互向量与实际值之间的点对点距离来实现。对于后一类方法，研究通常引入了 Dropout 技术，以鼓励模型恢复未观测到的反馈信息，从而防止过度拟合历史数据。这种方法从根本上将协作过滤建模为逆问题。

使用 Dropout 时，模型在每个训练迭代中随机“关闭”一些神经元，使它们的输出为零。这相当于在协同过滤的上下文中，随机地“关闭”或“遮蔽”某些用户与项目之间的关系，通过这种方式，Dropout 防止了模型过于依赖特定的用户或项目，类似于防止神经网络过度依赖特定的输入特征。在这个意义上，Dropout 的使用在某种程度上将协同过滤问题转化为一个逆问题，在协同过滤的情境下，观测到的结果是用户对项目的评分，而逆问题就是尝试推断出导致这些评分的用户和项目之间的关系。通过随机地关闭一些关系（使用Dropout），模型被迫更加通用地学习用户和项目之间的潜在模式，而不是过分拟合训练数据



Wang 等人将交互向量视为 DDPM 中的噪声潜变量，并通过学习基于 AE 的降噪器获得有竞争力的结果，此工作重点关注条件扩散模型（DM）的潜在应用。通过将逐步去噪与调节机制相结合，DM 在不同领域的各种 inverse problems 上取得了显着的成功，例如图像修复、加速 MRI、时间序列插补等



然而，标准高斯扩散可能不适合对隐式反馈进行建模，主要有两个原因：

1) 高斯扰动破坏了交互向量中的个性化信号，导致中间变量对推荐性能没有贡献
2) 各向同性噪声表未能考虑项目异质性，忽略了交互矩阵中存在的丰富结构信息



## Conditional Gaussian Diffusion

![image-20231206103442042](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231206103442042.png)

举个例子 
$$
q(x_{t}|x_{0}) = \mathcal{N}(x_{t};\alpha_{t}x_{0},\sigma_{t}^{2}I)
$$
且根据之前提到的，噪声程度是逐步递增的(0->1)，且 $\alpha_{t}$ 和 $\sigma_{t}$ 受限($\alpha_{t}^{2}+\sigma_{t}^{2}=1$)，鉴于分层生成，我们关注的是在两个任意时间步中，$x_s$ 和 $x_t$ 如何相互过渡

$$
\epsilon_{s}\,=\,\sqrt{\sigma_{s}^{2}-\sigma_{s|t}^{2}}\epsilon_{t}\,+\,\sigma_{s|t}\epsilon_{s|t}\\

q(x_{s}|x_{t},x_{0})\,=\,\,\,{\mathcal{N}}(x_{s};\alpha_{s}x_{0}+\sqrt{\sigma_{s}^{2}-\sigma_{s|t}^{2}}\epsilon_{t},\sigma_{s|t}^{2}I)\\
$$

## MC

数学定义：假设序列状态是$... X_{t-2},X_{t-1},X_{t},X_{t+1}...$，那么在 $X_{t+1}$ 时刻的状态的条件概率仅依赖于前一刻的状态$X_{t}$，即：
$$
P\left(X_{t+1}\mid\ldots X_{t-2},X_{t-1},X_{t}\right)=P\left(X_{t+1}\mid X_{t}\right)
$$

## Laplacian operator

平滑过程的一般思想是在（离散化的）黎曼流形上逐步交换信息
$$
\frac{\partial x}{\partial\tau}=\alpha\nabla^{2}x
$$
$x$ 的 Laplace 算子衡量 $x$ 与其邻域平均值之间的信息差异，$\alpha$ 捕获信息随时间 $\tau$ 交换的速率，对于交互信号
$$
\nabla^{2}=A-I
$$
其中 $A$ 是 item-item 图的归一化邻接矩阵
$$
x(\tau)=\exp\{-\tau\alpha(I-A)\}x(0)=U\exp\{-\tau\alpha(1-\lambda)\}\odot U^{\top}x(0),\quad\tau\geq0
$$
$U=[u_{1},...,u_{\mathcal{|I|}}]$ 是 $A$ 的单位特征向量矩阵，$\lambda=[\lambda_{1},...,\lambda_{\mathcal{|I|}}]$ 是降序排列的对应的特征值，$U$ 为正交矩阵, $1 − \lambda$ 表示图频率，这个公式表示在时间 $\tau$ 处的状态 $x(\tau)$ 由初始状态 $x(0)$ 经过演化算子得到，方程可以被解释为以不同速率 $\alpha(1-\lambda)$ 以指数衰减交互信号 $x$ 的每个高频分量

即现实世界中对于具有成千上万个节点的 item-item 图，这种方式存在问题，计算困难，一种解决方式是使用关于 $\tau$ 的一阶泰勒展开来近似方程中的连续时间滤波器
$$
\exp\{-\tau\alpha(I-A)\}=(1-\tau\alpha)I+\tau\alpha A+{\cal O}(\tau)
$$
......

最终获得一个前向滤波器序列，仅考虑 item-item Graph 上消息的一跳传播
$$
F_{t}=(1-\tau_{t}\alpha)I+\tau_{t}\alpha A,\quad t=0,1,2,\ldots,T
$$

## Item-Item Graph

如何构建 item-item 图的关键是衡量两个 item 节点之间的相似度
$$
D_{\mathcal U}=\mathrm{diagMat}(X1)\mathrm{~and~}D_{\mathcal I}=\mathrm{diagMat}(X^{\top}1) 
$$
一种是经过两次 LightGCN 卷积
$$
\mathbf{\tilde A}^{2}_{LGN}={\left[\begin{array}{l l}{O}&{D_{\mathcal{U}}}^{-\frac{1}{2}}X{D_{\mathcal{I}}}^{-\frac{1}{2}}\\ {D_{\mathcal{I}}}^{-\frac{1}{2}}X{D_{\mathcal{U}}}^{-\frac{1}{2}}&{O}\end{array}\right]}^{2}
$$
Fu 等人提出了一种更通用的形式
$$
A_{\beta,\gamma,\delta}=D_{\mathcal I}^{-\delta}X^{\top}D_{U}^{-\gamma}X D_{\mathcal I}^{-\beta}
$$
根据经验，$A_{\frac{1}{2},\frac{1}{2},\frac{1}{2}}$的效果相对于$A_{\frac{1}{2},1,\frac{1}{2}}$表现更好，但是这里只能聚合 2-hop 邻居，所以为了聚合高阶信息，提出新的改进,$U_{\beta,\gamma,\delta,d}$ 是 $A_{\beta,\gamma,\delta}$ 的 top-$d$ 特征值的特征向量矩阵，低通滤波器可以分解为
$$
U_{\beta,\gamma,\delta,d}U_{\beta,\gamma,\delta,d}^{\top}
$$

$$
A=\frac{1}{1+w}A_{\frac{1}{2},\frac{1}{2},\frac{1}{2}}/\|A_{\frac{1}{2},\frac{1}{2},\frac{1}{2}}\|_{2}+\frac{w}{1+w}U_{\frac{1}{2},\frac{1}{2},\frac{1}{2},d}U_{\frac{1}{2},\frac{1}{2},\frac{1}{2},d}^{\top}
$$

## Collaborative Forward Process

$$
x_{t}=F_{t}x_{0}+\sigma_{t}\epsilon_{t},\epsilon_{t}\sim{\mathcal N}(0,I),t=0,1,2,...,T,\\
x_{t}=x_{0}+\sigma_{t}\epsilon_{t},\epsilon_{t}\sim{\mathcal N}(0,I),t=0,1,2,...,T,
$$

最主要的区别在于 $x_{0}$ 前的系数
$$
F_{t}=\alpha_tI\\
F_{t}=U[(1-\tau_{t}\alpha){\bf1}+\tau_{t}\alpha\lambda]\odot U^{\top},\quad t=0,1,2,...,T.
$$
将 $F_{t}$ 带入原式子
$$
\tilde{x}_{t}=\left[(1-\tau_{t}\alpha){\bf1}+\tau_{t}\alpha\lambda\right]\odot\tilde{x}_{0}+\sigma_{t}\tilde{\epsilon}_{t}, \\
\tilde{\epsilon}_{t}\sim{\mathcal N}(0,I)\\
t=0,1,2,...,T.
$$

## Personalized Reverse Process

前向过程
$$
q(x_{t}|x_{0})={\cal N}(x_{t};F_{t}x_{0},\sigma_{t}^{2}{I}) 
$$
后向过程
$$
p_{\theta}({x}_{0}|{x}_{T},\,c)=\prod_{t=1}^{T}p_{\theta}({x}_{t-1}|{x}_{t},\,c)
$$
![image-20231208200811492](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231208200811492.png)

输入交互矩阵 $X$，输出去噪器的参数 $\theta$

1. 重复
2. 从集合 $\mathcal U$ 中采样一个 $u$，将 $x_u$ 赋值给 $x_{0}$
3. 随机屏蔽 $x_{0}$ 获得 $c$
4. 从均匀分布采样得到 $t$，基于这个 $t$ 获得一个正态分布得 $x_{t}$
5. 进行梯度步骤，即训练 ${\hat x}_{\theta}$ 

![image-20231208200819743](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231208200819743.png)

输入去噪器的参数 $\theta$ 和历史交互 $x_u$

1. 将 $x_u$ 赋值给 $c$，$F_{T}c$ 赋值给 $x_{u}$
2. 进行 T 次去噪过程



## DDIM

Preference Denoiser 的设计
$$
\hat{x}_{\theta}(x_{t},c,t)=W^{\top}\mathrm{Dropout}\left(\theta_{1}(t)W\frac{x_{t}}{||c||}+\theta_{2}(t)W\frac{c}{||c||}\right)
$$
$\mathbf{W}\in\mathbb{R}^{d\times|{\mathcal I}|}$ 是编码器和解码器共享权重矩阵，这里是一组 item embeddings，$\theta_{1}(t),\theta_{2}(t)$ 是基于正弦基的时间步长 t 的两个可学习标量函数，并在解码器层之前使用额外的 dropout 以防止过拟合，将算法二整理为
$$
x_{t-1}=x_{t}-\frac{1}{T}(I-A)\hat{x}_{\theta}(x_{t},c,t)
$$
所实现的反向过程迭代地消除了初始交互信号和平滑交互信号之间的预测差异