**CKER 结构**

![image-20230814101042261](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230814101042261.png)



**知识增强推荐的目的**

是从交互数据 ${\cal O}+$ 和知识图 ${\cal G}_k$ 中学习 user和 item 的表示，从而预测每个用户采用候选项目的可能性有多大；那么排名前K位的物品就构成了对用户的推荐列表



**协同图卷积网络**

为了同时利用 user-item 交互中反映的 item 之间的交互关系和知识图引入的 item 知识，我们提出CGCN来学习项目表示，它由两个通道组成，即 interaction graph propagation 和 the knowledge graph propagation



**Interaction Graph Propagation**

interaction graph (IG)
$$
{\cal G}_{I}=\{{\cal I},{\mathcal E}_{I}\}
$$
$\cal I$ 表示节点,也就是全部 item，${\mathcal E}_{I}$ 表示边，每条边 $
\left(i_{\varepsilon},i_{\kappa}\right)\in{\mathcal{E}}_{I} 
$ 表示 $i_{\varepsilon},i_{\kappa}$ 是由同一个用户交互的 item，根据边缘权重应用最大采样来选择MI(GI 中每个节点的邻居个数)最相关的项目作为每个项目的最终邻居，从而滤除用户不确定行为模式引入的噪声

构建交互图后，从 ${\cal G}_{I}$ 中每个项目的邻居传播信息，以利用项目之间的交互关系来更新项目表示。采用 LightGCN 来进行信息传播
$$
\hat{\bf e}_{i}^{l}=\frac{1}{|{\cal N}_{i}^{I}|}\sum_{\epsilon\in{\cal N}_{i}^{I}}\hat{\bf e}_{\epsilon}^{l-1}
$$
Knowledge Graph Propagation

triplet $\left(i,r,v\right)\in {\cal G}_K$，通过依赖边权重来应用最大采样来选择MK最相关的实体作为每个实体的最终邻居，以避免引入偏差。同时，在 KG 中，每个尾部实体与不同的关系配对时具有不同的语义。因此，通过聚合 KG 中相应的关系尾对来获取每个实体的邻居信息
$$
\tilde{\bf e}_{i}^{l}=\frac{1}{|{\cal N}_{i}^{K}|}\sum_{(r,v)\in{\cal N}_{i}^{K}}{\bf e}_{r}\odot\tilde{\mathbf e}_{v}^{l-1}
$$
Multi-Layer Graph Convolutions
$$
{\bf e}_{i}^{l}=\hat{\bf e}_{i}^{l}+\tilde{\bf e}_{i}^{l}
$$
通过 sum pooling 将它们组合在一起，以获得项目 $i$ 的最终潜在向量
$$
{\mathbf e}_{i}^{l}=CGCN({\bf e}_{i}^{l-1},{\cal N}_{i}^{I},{\cal N}_{i}^{K})
$$
可以用以上公式简要概括 CGCN 的步骤
$$
{\bf e}_{i}^{*}\,=\,{\bf e}_{i}^{0}\,+\,{\bf e}_{i}^{1}\,+\,\cdot\,\cdot\,+\,{\bf e}_{i}^{L}
$$


**用户偏好生成**

user-item bipartite graph $
{\mathcal{G}}_{U}=\{\mathcal U\cup\mathcal{I},{\mathcal{E}}_{U}\} 
$
$$
\hat{\bf e}_{u}^{l}=\frac{1}{|{\cal N}_{u}^{U}|}\sum_{i\in{\cal N}_{u}^{U}}\hat{\bf e}_{i}^{l-1}
$$

$$
\hat{\bf e}_{u}^{*}=\hat{\bf e}_{u}^{0}+\hat{\bf e}_{u}^{1}+\cdot\cdot\cdot+\hat{\bf e}_{u}^{L}
$$

类似地，通过聚合多层知识图卷积生成的项目表示来获得知识感知的用户偏好
$$
\tilde{\bf{e}}_{u
}^{l}=\frac{1}{|{\cal N}_{u}^{U}|}\sum_{i\in{\cal N}_{u}^{U}}\tilde{\bf{e}}_{i}^{l-1}
$$

$$
\tilde{\bf e}_{u}^{*}=\tilde{\bf e}_{u}^{0}+\tilde{\bf e}_{u}^{1}+\cdot\cdot\cdot+\tilde{\bf e}_{u}^{L}
$$

$$
\mathbf{e}_{\mathcal{u}}^{*}=\hat{\mathbf{e}}_{\mathcal{u}}^{*}+\tilde{\mathbf{e}}_{\mathcal{u}}^{*}
$$



**监督学习**
$$
\hat{y}_{u i}={\bf e}_{u}^{\ast\top}{\bf e}_{i}^{\ast}
$$

$$
{\mathcal{L}}_{m a i n}=\sum_{(u,i,j)\in{\mathcal{O}}}-\log(\sigma({\hat{y}}_{u i}-{\hat{y}}_{u j}))
$$



**自监督学习**

假设当前小批量由 $N$ 个用户组成，那么用户 $u_{\varepsilon}$ 的知识感知偏好应该比小批量中其他 $N-1$ 个用户更类似于用户 $u_{\varepsilon}$ 的交互感知偏好

positive pair: $(\tilde{\bf e}_{u_\varepsilon}^{*},\hat{\bf e}_{u_\varepsilon}^{*}) $

negative pair:$
(\tilde{\bf e}_{u_{\epsilon}}^{*},\hat{\bf e}_{u_{\kappa}}^{*})|\kappa\,=\,1,...,\epsilon\,-\,1,\epsilon\,+\,1,...,N]) $


$$
{\mathcal{L}}_{s s l}={\frac{\exp(\lambda\sin({\tilde{\mathbf{e}}}_{u_{\epsilon}}^{*},{\hat{\mathbf{e}}}_{u_{\epsilon}}^{*}))}{\exp(\lambda\sin({\tilde{\mathbf{e}}}_{u_{\epsilon}}^{*},{\hat{\mathbf{e}}}_{u_{\epsilon}}^{*}))+\sum_{\mathbf{\kappa}=1,\mathbf{\kappa}\neq \epsilon}^{N}\exp(\lambda\sin({\tilde{\mathbf{e}}}_{u_{\epsilon}}^{*},{\hat{\mathbf{e}}}_{u_{\kappa}}^{*}))}}
$$

$$
\mathrm{sim}\big({\bf u},{\bf v}\big)=\mathrm{cos}\big({\bf u},{\bf v}\big)={\bf u}^{\top
}{\bf v}/||{\bf u}||||{\bf v}||
$$

$$
{\cal L}={\cal L}_{m a i n}+\alpha{\cal L}_{s s l}
$$



**对比基线**

KGAT