## Aggregate Neighbors

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230831153758696.png" alt="image-20230831153758696" style="zoom:80%;" />

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913164708546.png" alt="image-20230913164708546" style="zoom:50%;" />

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913164726420.png" alt="image-20230913164726420" style="zoom:50%;" />

左乘 $A$ 是求和，再乘 $D^{-1}$ 是求平均的过程

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913164934729.png" alt="image-20230913164934729" style="zoom:50%;" />

但是这种方式只考虑了节点自己的 degree，没有考虑对方节点，所以做出改进

![image-20230913165315222](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913165315222.png)

同时为了保持”值“的稳定性，不影响输入向量的大小，得到 Normalized Diffusion/Adjacency Matrix $\tilde A$

![image-20230913165501453](C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913165501453.png)

<img src="C:\Users\Asus\AppData\Roaming\Typora\typora-user-images\image-20230913172204049.png" alt="image-20230913172204049" style="zoom:80%;" />