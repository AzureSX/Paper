## Abstract

random-dropping 不可取

将潜在邻居纳入对比对中

## INTRODUCTION

现有的神经图协同过滤方法仍然面临两个主要问题：数据通常是稀疏或嘈杂

高阶关系或约束（例如用户或项目相似性）无法显式地用于丰富图信息

节点级关系：（1）结构邻居是指通过高阶路径结构连接的节点 （2）语义邻居指语义相似的邻居可能无法在图上直接到达

## METHODOLOGY

**图协同过滤主干 Graph Collaborative Filtering BackBone**

和 LightGCN 一摸一样



**Contrastive Learning with Structural Neighbors**

交互图G是二分图，从 GNN 模型的偶数层（例如 2、4、6）输出中获得同质邻域的表示

将用户本身的嵌入和偶数层 GNN 的相应输出的嵌入视为正对

![image-20230927194014325](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194014325.png)

![image-20230927194150291](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194150291.png)

![image-20230927194156545](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194156545.png)

**Contrastive Learning with Semantic Neighbors**

图上无法到达但具有相似特征（项目节点）或偏好（用户节点）的节点

受先前作品[16]的启发，受先前作品[16]的启发，受先前作品[16]的启发,通过学习每个用户和项目的潜在原型来识别语义

邻居，相似的用户/项目往往落在相邻的嵌入空间中，并且原型是代表一组语义邻居的簇的中心

对用户和项目的嵌入应用聚类算法，以获得用户或项目的原型

![image-20230927194818890](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194818890.png)

![image-20230927194847232](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194847232.png)

![image-20230927194856554](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194856554.png)

![image-20230927194909210](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194909210.png)

**Optimization**

![image-20230927194956235](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927194956235.png)

使用 EM 算法优化 L𝑃

![image-20230927195037104](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927195037104.png)

**创新和困难**

这是利用结构和语义邻居进行图协同过滤的首次尝试，没有引入额外的图构造或邻域迭代

应用原型学习技术来捕获语义信息



**实验**

![image-20230927195935790](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927195935790.png)

LightGCN 一直是最好的

![image-20230927195959716](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230927195959716.png)

越均匀越好