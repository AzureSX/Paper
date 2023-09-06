# Preliminary

## 主要的数学符号

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230824214430851.png" alt="image-20230824214430851" style="zoom:75%;" />

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230824214451286.png" alt="image-20230824214451286" style="zoom:75%;" />



# Methodology

分为3个部分，分别是 1)监督过程 2)自监督过程 3)多任务训练

$S\;=\;\{(v,e)\;\;|\;v\;\in\;{\mathcal{V}},e\;\in\;{\mathcal{E}}\}$ item-entity 关联矩阵



![image-20230824215439518](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230824215439518.png)



## Supervised Process

**Collaboration Propagation**
$$
{\mathcal{E}}_{u}^{0}=\{e\mid(v,e)\in S\;\,{\mathrm{and}}\;\;v\in{\mathcal{V}}_{u}\}
$$
用 $\mathcal{V}_u$ 表示用户 u，表示用户 u 交互过的项目
$$
{\mathcal U}_{v}=\{u\mid y_{u v}=1\}
$$

$$
{\mathcal{E}}_{v}^{0}=\{e\mid(v^{\prime},e)\in S\ a n d\ \ v^{\prime}\in{\mathcal{V}}_{u^{\prime}}\ a n d\ \ u^{\prime}\in{\mathcal{U}}_{v}\}
$$

item v 是由项目交互过的 user 所交互的其他的 items 所对应的 entities 所决定的

**Knowledge Propagation**
$$
{\mathcal{E}}_{x}^{m}=\{t\mid(h,r,t)\in\mathbf{G},h\in{\mathcal{E}}_{x}^{m-1}\},m=1,2,...L.
$$
${\mathcal{E}}_{x}^{m}$ 表示 user or item 第 m 层的实体集
$$
{\mathcal T}_{x}^{m}=\{(h,r,t)\mid(h,r,t)\in{\bf G},h\in{\mathcal E}_{x}^{m-1}\},m=1,2,...L.
$$
此公式需要利用上一个公式的结果，与上一个公式不同，这个公式计算的是三元组

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230827101247956.png" alt="image-20230827101247956" style="zoom:50%;" />
$$
\beta_{i}=\mu(e_{i}^{h},e_{i}^{r})e_{i}^{t}
$$

$$
\mu(e_{i}^{h},e_{i}^{r})=\sigma(W_{2}R e L u(W_{1}\rho_{0}+b_{1})+b_{2})
$$

归一化
$$
\mu(e_{i}^{h},e_{i}^{r})=\frac{e x p(\mu(e_{i}^{h},e_{i}^{r}))}{\sum(h^{\prime},r^{\prime},t^{\prime})\in\mathcal{T}_{x}^{m}e x p(\mu(e_{i}^{h^{\prime}},e_{i}^{r^{\prime}}))}
$$
用第 m 层三元组计算出的 x 的嵌入
$$
e_{x}^{(m)}=\sum_{i=1}^{\lfloor {\mathcal E}_{x}^{(m)}\rfloor}{{\beta}}_{i}
$$
在 KG 中传播 L 层
$$
\mathbb{E}_{x}=\{e_{x}^{(0)},e_{x}^{(1)},...,e_{x}^{(L)}\}
$$
**Prediction and Loss Function**

最终的 embedding
$$
e_{x}=\sigma\,(W_{a}(e_{x}^{(0)}\|e_{x}^{(1)}\|...\|e_{x}^{(L)})+b_{a})
$$

$$
\hat{y}_{u v}=e_{u}^{\top}\,e_{v}
$$

$$
{\mathcal{L}}_{m a i n}=\sum_{(u,i,j)\in{\mathcal{Z}}}-l o g(\sigma({\hat{y}}_{u i}-{\hat{y}}_{u j}))
$$

## Self-supervised Process

**Data Augmentation**

仍然是随机删除
$$
{\mathbf Y}^{i}=({\mathbf Y}\odot M_{i}),i=1,2
$$
然后按照第一步的传播过程再次构建相关集合
$$
{\mathcal E}_{u i}^{0}=\{e\mid(v,e)\in S\ \,a n d\ \ v\in\{v\mid y_{u v}^{i}=1\}\},i=1,2.
$$

$$
{\mathcal U}_{ui}=\{u\mid y_{u v}^{i}=1\},i=1,2.
$$

$$
{\mathcal E}_{v i}^{0}\,=\,\{e\mid(v,e)\in S\;\,a n d\,\,\,v\,\in\{v\mid y_{u v}^{i}=1,u\in {\mathcal U}_{v i}\},i=1,2
$$

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230827141811538.png" alt="image-20230827141811538" style="zoom:50%;" />
$$
\mathbb{E}_{x1}=\{e_{x1}^{(0)},e_{x1}^{(1)},...,e_{x1}^{(L)}\}
$$

$$
\mathbb{E}_{x2}=\{e_{x2}^{(0)},e_{x2}^{(1)},...,e_{x}^{(L)}\}
$$

$$
e_{x}=\sigma\,(W_{a}(e_{x}^{(0)}\|e_{x}^{(1)}\|...\|e_{x}^{(L)})+b_{a})
$$

**Contrastive Learning**
$$
S S{\cal L}_{u s e r}=\sum_{u\in{\cal U}}-l o g\frac{e x p(s(e_{u1},e_{u2})/\tau)}{\sum_{w\in{\cal U}}e x p(s(e_{u1},e_{w2})/\tau)}
$$

$$
{\mathcal{L}}_{s s l}=S S{\mathcal{L}}_{u s e r}+S S{\mathcal{L}}_{i t e m}
$$

## Multi-task Training

$$
{\mathcal{L}}={\mathcal{L}}_{m a i n}+\lambda_{1}{\mathcal{L}}_{s s l}+\lambda_{2}\vert\Theta\vert_{2}^{2}
$$



# Experiment