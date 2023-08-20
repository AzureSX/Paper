**传统 GNN 存在的问题**

利用图神经网络（GNN）捕获KG中的信息并进一步应用于RS中仍然存在问题，因为它无法从多个角度查看每个项目的属性

**User-Oriented Relation Attention**
$$
{\mathbf n}=\sum_{e\in{\cal N}(v)}\widetilde{\pi}_{r_{v,e}}^{u}\mathrm{\mathbf e}
$$

$$
\widetilde{\pi}_{r_{v,\,e}}^{u}=\pi({\bf v},{\bf e})={\frac{\exp(\pi_{r_{v,\,e}}^{u})}{\sum_{e^{\prime}\in {\cal N}(v)}\exp(\pi_{r_{v,\,e^\prime}}^{u})}}
$$

$$
\pi_{r_{v,\,e}}^{u}=W_{r}(\mathrm{concat}([{\bf u},{\bf r},{\bf v}]))+{\bf b}_{r}
$$

**User-Oriented Entity Projection**
$$
\begin{array}{l}{{\mathbf{\widetilde{e}}=W_{e}(\mathbf{e}+\mathbf{u})+\mathbf{b}_{e}}}\\ {{\mathbf{\widetilde{e}}=\sigma(W_{e}(\mathbf{e}+\mathbf{u})+\mathbf{b}_{e})}}\end{array}
$$
**KG-Enhanced User Representation**