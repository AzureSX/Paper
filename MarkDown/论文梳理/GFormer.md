## INTRODUCTION

盲目破坏图拓扑结构可能会导致用户和项目之间关键关系的丢失，例如独特的用户交互模式或长尾项目的有限标签

需要设计的增强器具有不变的基本原理

从对齐局部级和全局级嵌入以进行增强的角度来看，一些研究通过各种信息聚合技术获得语义相关的子图表示，例如 HCCF 中基于超图的消息传递和 NCL 中基于 EM 算法的节点聚类

增强的质量可能会受到手动构建的超图结构和用户集群设置的影响，此外，这些手动设计的对比方法很容易被常见的噪音所误导

受最近成功的掩码自动编码（MAE）技术在推进自监督学习方面的推动[3,4,7]，这项工作从生成自监督增强和理性感知不变表

示学习的角度探讨了上述问题 Masked autoencoders

## METHODOLOGY

**Graph Invariant Rationale Learning**

Recently, rationalization learning techniques have been introduced into graph representation learning by discovering **invariant rationales for important graph structural information** to benefit downstream graph mining tasks [11, 12, 26]. I

在基于图的 CF 场景中，我们的不变原理发现方案旨在找到图结构的子集，该子集最好地指导下游推荐任务的自我监督增强和

合理化

![image-20230928091018109](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230928091018109.png)

**Graph Collaborative Rationale Discovery**

![image-20230929103344024](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230929103344024.png)

![image-20230929103350208](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230929103350208.png)

**Global Topology Information Injection**