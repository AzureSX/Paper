![image-20231218192755137](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231218192755137.png)

![image-20231218193004961](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20231218193004961.png)

# 拓扑级自适应增强

$$
t\left({\mathcal{G}}\right)=\left(\mathcal{V},m\circ\mathcal{E}\right),
$$

$m\in\{0,1\}^{|\mathcal{E}|}$ 显而易见是 masking vector，$m_{i}=B e r n\left(p_{u i}^{e}\right) $，其中 $p_{u i}^{e}$ 是对一条边(edge $(u,i)\in\mathcal{E}$)进行采

样的概率，这个概率反映了边的重要性。$\varphi_{c}\left(\cdot\right):{\mathcal V}\rightarrow\mathbb{R}^{+} $，给定两个相邻节点的中心性定义边的中心性 $w_{u i}^{e}$ ,
$$
w_{u i}^{e}\;=\;\left(\varphi_{c}\,(u)+\varphi_{c}\,(i)\right)/2
$$
由于数据交互不平衡,利用 $s_{u i}^{e}=\log{w_{u i}^{e}}$
$$
p_{u i}^{e}=\frac{s_{u i}^{e}-s_{m i n}^{e}}{s_{m a x}^{e}-s_{m i n}^{e}}\cdot p_{1}^{e}+p_{2}^{e}
$$
再使用 truncation 函数使最终的值落到 0-1 区间
$$
p_{u i}^{e}=\mathrm{max}\,(p_{u i}^{e},0)，p_{u i}^{e}=\mathrm{min}\,(p_{u i}^{e},1)，
$$


# 特征级自适应增强

LightGCN 得到的特征嵌入表示
$$
{\bf X}^{\left(\ell+1\right)}=\left({{\bf D}}^{-\frac{1}{2}}{\bf A}{\bf D}^{-\frac{1}{2}}\right){\bf X}^{\left(\ell\right)}
$$

$$
{\bf X}^{(\ell+1)}=\left({\bf D}^{-\frac{1}{2}}{\bf A}{\bf D}^{-\frac{1}{2}}\right){\bf X}^{(\ell)}+{\bf\Delta}^{(\ell+1)}\odot\tilde{m}
$$

$\tilde m\in\{0,1\}^{d}$ 是长度为 $d$ 的 masking vector，$\tilde m_{j}=B e r n\left(p_{j}^{f}\right) $，其中 $p_{j}^{f}$ 是向第 $j$ 维添加噪声扰动的概率
$$
{\bf\Delta}\in{\mathbb{R}}^{N\times d}
$$
这坨看代码怎么写的，文字有点抽象