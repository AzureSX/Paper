**Task Formulation**

Input: 用户项目交互数据 ${\mathcal G}_u = {\{\mathcal{V,E}\}}$，项目知识图数据 ${\mathcal G}_k = {\{(h,r,t)\}}$

Output: $
\mathcal{F}=\left(u,v\right|\mathcal{G}_{u},\mathcal{G}_{k},\mathcal{\mathbf{\Theta}}) 
$



$p_{u,i}$ 表示丢弃用户 $u$ 和项目 $i$ 之间的交互边的估计概率

$w_{u,i}$ 表示项目 $i$ 对用户 $u$ 的影响程度，其与 $c_i$ 对应的结构一致性得分成正比

$p^{\prime}_{u,i}$ 表示用截断概率 $p_𝜏$ 对 $w_{u,i}$ 进行最小-最大归一化，以减轻低值效应

$u_{p^{\prime}}$ 表示平均值

$p_a$ 控制基于平均值的影响力的强度



以概率 $p_{u,i}$，进一步基于伯努利分布生成两个掩蔽向量 $\mathbf M_u^1,\mathbf M_u^2 \in \{0,1\}$。之后，将 $\mathbf M_u^1,\mathbf M_u^2 \in \{0,1\}$ 应用于用户-项目交互图 $\cal G_$