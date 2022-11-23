# 博客链接

[(2 封私信 / 3 条消息) 朱勇椿 - 知乎 (zhihu.com)](https://www.zhihu.com/people/zhu-yong-chun-88/posts)

[KDD'22推荐系统论文梳理 (24篇研究&36篇应用论文) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/550813024)





# KDD 2022

[Addressing Unmeasured Confounder for Recommendation with Sensitivity Analysis](https://dl.acm.org/doi/10.1145/3534678.3539240)

- Sihao Ding
- Peng Wu
- Fuli Feng
- Yitong Wang
- Xiangnan He
- Yong Liao
- Yongdong Zhang

Recommender systems should answer the intervention question "if recommending an item to a user, what would the feedback be", calling for estimating the causal effect of a recommendation on user feedback. Generally, this requires blocking the effect of confounders that simultaneously affect the recommendation and feedback. To mitigate the confounding bias, a strategy is incorporating propensity into model learning. However, existing methods forgo possible unmeasured confounders (e.g., user financial status), which can result in biased propensities and hurt recommendation performance. This work combats the risk of unmeasured confounders in recommender systems.

Towards this end, we propose Robust Deconfounder (RD) that accounts for the effect of unmeasured confounders on propensities, under the mild assumption that the effect is bounded. It estimates the bound with sensitivity analysis, learning a recommender model robust to unmeasured confounders within the bound by adversarial learning. However, pursuing robustness within a bound may restrict model accuracy. To avoid the trade-off between robustness and accuracy, we further propose Benchmarked RD (BRD) that incorporates a pre-trained model into the learning as the benchmark. Theoretical analyses prove the stronger robustness of our methods compared to existing propensity-based deconfounders, and also prove the no-harm property of BRD. Our methods are applicable to any propensity-based estimators, where we select three representative ones: IPS, Doubly Robust, and AutoDebias. We conduct experiments on three real-world datasets to demonstrate the effectiveness of our methods.



[PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions](https://dl.acm.org/doi/10.1145/3534678.3539432)

- Ehsan Gholami
- Mohammad Motamedi
- Ashwin Aravindakshan

The emerging meta- and multi-verse landscape is yet another step towards the more prevalent use of already ubiquitous online markets. In such markets, recommender systems play critical roles by offering items of interest to the users, thereby narrowing down a vast search space that comprises hundreds of thousands of products. Recommender systems are usually designed to learn common user behaviors and rely on them for inference. This approach, while effective, is oblivious to subtle idiosyncrasies that differentiate humans from each other. Focusing on this observation, we propose an architecture that relies on common patterns as well as individual behaviors to tailor its recommendations for each person. Simulations under a controlled environment show that our proposed model learns interpretable personalized user behaviors. Our empirical results on Nielsen Consumer Panel dataset indicate that the proposed approach achieves up to 27.9% performance improvement compared to the state-of-the-art.



[Feature Overcorrelation in Deep Graph Neural Networks: A New Perspective](https://dl.acm.org/doi/10.1145/3534678.3539445)

- Wei Jin
- Xiaorui Liu
- Yao Ma
- Charu Aggarwal
- Jiliang Tang

Recent years have witnessed remarkable success achieved by graph neural networks (GNNs) in many real-world applications such as recommendation and drug discovery. Despite the success, oversmoothing has been identified as one of the key issues which limit the performance of deep GNNs. It indicates that the learned node representations are highly indistinguishable due to the stacked aggregators. In this paper, we propose a new perspective to look at the performance degradation of deep GNNs, i.e., feature overcorrelation. Through empirical and theoretical study on this matter, we demonstrate the existence of feature overcorrelation in deeper GNNs and reveal potential reasons leading to this issue. To reduce the feature correlation, we propose a general framework DeCorr which can encourage GNNs to encode less redundant information. Extensive experiments have demonstrated that DeCorr can help enable deeper GNNs and is complementary to existing techniques tackling the oversmoothing issue



[User-Event Graph Embedding Learning for Context-Aware Recommendation](https://dl.acm.org/doi/10.1145/3534678.3539458)

- Dugang Liu
- Mingkai He
- Jinwei Luo
- Jiangxu Lin
- Meng Wang
- Xiaolian Zhang
- Weike Pan
- Zhong Ming

Most methods for context-aware recommendation focus on improving the feature interaction layer, but overlook the embedding layer. However, an embedding layer with random initialization often suffers in practice from the sparsity of the contextual features, as well as the interactions between the users (or items) and context. In this paper, we propose a novel user-event graph embedding learning (UEG-EL) framework to address these two sparsity challenges. Specifically, our UEG-EL contains three modules: 1) a graph construction module is used to obtain a user-event graph containing nodes for users, intents and items, where the intent nodes are generated by applying intent node attention (INA) on nodes of the contextual features; 2) a user-event collaborative graph convolution module is designed to obtain the refined embeddings of all features by executing a new convolution strategy on the user-event graph, where each intent node acts as a hub to efficiently propagate the information among different features; 3) a recommendation module is equipped to integrate some existing context-aware recommendation model, where the feature embeddings are directly initialized with the obtained refined embeddings. Moreover, we identify a unique challenge of the basic framework, that is, the contextual features associated with too many instances may suffer from noise when aggregating the information. We thus further propose a simple but effective variant, i.e., UEG-EL-V, in order to prune the information propagation of the contextual features. Finally, we conduct extensive experiments on three public datasets to verify the effectiveness and compatibility of our UEG-EL and its variant.



[Joint Knowledge Graph Completion and Question Answering](https://dl.acm.org/doi/10.1145/3534678.3539289)

- Lihui Liu
- Boxin Du
- Jiejun Xu
- Yinglong Xia
- Hanghang Tong

Knowledge graph reasoning plays a pivotal role in many real-world applications, such as network alignment, computational fact-checking, recommendation, and many more. Among these applications, knowledge graph completion (KGC) and multi-hop question answering over knowledge graph (Multi-hop KGQA) are two representative reasoning tasks. In the vast majority of the existing works, the two tasks are considered separately with different models or algorithms. However, we envision that KGC and Multi-hop KGQA are closely related to each other. Therefore, the two tasks will benefit from each other if they are approached adequately. In this work, we propose a neural model named BiNet to jointly handle KGC and multi-hop KGQA, and formulate it as a multi-task learning problem. Specifically, our proposed model leverages a shared embedding space and an answer scoring module, which allows the two tasks to automatically share latent features and learn the interactions between natural language question decoder and answer scoring module. Compared to the existing methods, the proposed BiNet model addresses both multi-hop KGQA and KGC tasks simultaneously with superior performance. Experiment results show that BiNet outperforms state-of-the-art methods on a wide range of KGQA and KGC benchmark datasets.



[Practical Counterfactual Policy Learning for Top-K Recommendations](https://dl.acm.org/doi/10.1145/3534678.3539295)

- Yaxu Liu
- Jui-Nan Yen
- Bowen Yuan
- Rundong Shi
- Peng Yan
- Chih-Jen Lin

For building recommender systems, a critical task is to learn a policy with collected feedback (e.g., ratings, clicks) to decide which items to be recommended to users. However, it has been shown that the selection bias in the collected feedback leads to biased learning and thus a sub-optimal policy. To deal with this issue, counterfactual learning has received much attention, where existing approaches can be categorized as either value learning or policy learning approaches. This work studies policy learning approaches for top-K recommendations with a large item space and points out several difficulties related to importance weight explosion, observation insufficiency, and training efficiency. A practical framework for policy learning is then proposed to overcome these difficulties. Our experiments confirm the effectiveness and efficiency of the proposed framework.



[Extracting Relevant Information from User's Utterances in Conversational Search and Recommendation](https://dl.acm.org/doi/10.1145/3534678.3539471)

- Ali Montazeralghaem
- James Allan

Conversational search and recommendation systems can ask clarifying questions through the conversation and collect valuable information from users. However, an important question remains: how can we extract relevant information from the user's utterances and use it in the retrieval or recommendation in the next turn of the conversation? Utilizing relevant information from users' utterances leads the system to better results at the end of the conversation. In this paper, we propose a model based on reinforcement learning, namely RelInCo, which takes the user's utterances and the context of the conversation and classifies each word in the user's utterances as belonging to the relevant or non-relevant class. RelInCo uses two Actors: 1) Arrangement-Actor, which finds the most relevant order of words in user's utterances, and 2) Selector-Actor, which determines which words, in the order provided by the arrangement Actor, can bring the system closer to the target of the conversation. In this way, we can find relevant information in the user's utterance and use it in the conversation. The objective function in our model is designed in such a way that it can maximize any desired retrieval and recommendation metrics (i.e., the ultimate



[Aligning Dual Disentangled User Representations from Ratings and Textual Content](https://dl.acm.org/doi/10.1145/3534678.3539474)

- Nhu-Thuat Tran
- Hady W. Lauw

Classical recommendation methods typically render user representation as a single vector in latent space. Oftentimes, a user's interactions with items are influenced by several hidden factors. To better uncover these hidden factors, we seek disentangled representations. Existing disentanglement methods for recommendations are mainly concerned with user-item interactions alone. To further improve not only the effectiveness of recommendations but also the interpretability of the representations, we propose to learn a second set of disentangled user representations from textual content and to align the two sets of representations with one another. The purpose of this coupling is two-fold. For one benefit, we leverage textual content to resolve sparsity of user-item interactions, leading to higher recommendation accuracy. For another benefit, by regularizing factors learned from user-item interactions with factors learned from textual content, we map uninterpretable dimensions from user representation into words. An attention-based alignment is introduced to align and enrich hidden factors representations. A series of experiments conducted on four real-world datasets show the efficacy of our methods in improving recommendation quality.



[Comprehensive Fair Meta-learned Recommender System](https://dl.acm.org/doi/10.1145/3534678.3539269)

- Tianxin Wei
- Jingrui He

In recommender systems, one common challenge is the cold-start problem, where interactions are very limited for fresh users in the systems. To address this challenge, recently, many works introduce the meta-optimization idea into the recommendation scenarios, i.e. learning to learn the user preference by only a few past interaction items. The core idea is to learn global shared meta-initialization parameters for all users and rapidly adapt them into local parameters for each user respectively. They aim at deriving general knowledge across preference learning of various users, so as to rapidly adapt to the future new user with the learned prior and a small amount of training data. However, previous works have shown that recommender systems are generally vulnerable to bias and unfairness. Despite the success of meta-learning at improving the recommendation performance with cold-start, the fairness issues are largely overlooked.

In this paper, we propose a comprehensive fair meta-learning framework, named CLOVER, for ensuring the fairness of meta-learned recommendation models. We systematically study three kinds of fairness - individual fairness, counterfactual fairness, and group fairness in the recommender systems, and propose to satisfy all three kinds via a multi-task adversarial learning scheme. Our framework offers a generic training paradigm that is applicable to different meta-learned recommender systems. We demonstrate the effectiveness of CLOVER on the representative meta-learned user preference estimator on three real-world data sets. Empirical results show that CLOVER achieves comprehensive fairness without deteriorating the overall cold-start recommendation performance.



[Improving Social Network Embedding via New Second-Order Continuous Graph Neural Networks](https://dl.acm.org/doi/10.1145/3534678.3539415)

- Yanfu Zhang
- Shangqian Gao
- Jian Pei
- Heng Huang

Graph neural networks (GNN) are powerful tools in many web research problems. However, existing GNNs are not fully suitable for many real-world web applications. For example, over-smoothing may affect personalized recommendations and the lack of an explanation for the GNN prediction hind the understanding of many business scenarios. To address these problems, in this paper, we propose a new second-order continuous GNN which naturally avoids over-smoothing and enjoys better interpretability. There is some research interest in continuous graph neural networks inspired by the recent success of neural ordinary differential equations (ODEs). However, there are some remaining problems w.r.t. the prevailing first-order continuous GNN frameworks. Firstly, augmenting node features is an essential, however heuristic step for the numerical stability of current frameworks; secondly, first-order methods characterize a diffusion process, in which the over-smoothing effect w.r.t. node representations are intrinsic; and thirdly, there are some difficulties to integrate the topology of graphs into the ODEs. Therefore, we propose a framework employing second-order graph neural networks, which usually learn a less stiff transformation than the first-order counterpart. Our method can also be viewed as a coupled first-order model, which is easy to implement. We propose a semi-model-agnostic method based on our model to enhance the prediction explanation using high-order information. We construct an analog between continuous GNNs and some famous partial differential equations and discuss some properties of the first and second-order models. Extensive experiments demonstrate the effectiveness of our proposed method, and the results outperform related baselines.



# SIGIR 2022 POI

**Learning Graph-based Disentangled Representations for Next POI Recommendation**

**GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation**

**Empowering Next POI Recommendation with Multi-Relational Modeling**

**Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation**





# SIGIP 2022 KG

**KETCH: Knowledge Graph Enhanced Thread Recommendation in Healthcare Forums**

**Post Processing Recommender Systems with Knowledge Graphs for Recency, Popularity, and Diversity of Explanations**

**Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion**

**Meta-Knowledge Transfer for Inductive Knowledge Graph Embedding**

**Re-thinking Knowledge Graph Completion Evaluation from an Information Retrieval Perspective**



# IJCAI 2022 Rec

**RecipeRec: A Heterogeneous Graph Learning Model for Recipe Recommendation**

**Self-supervised Graph Neural Networks for Multi-behavior Recommendation**

**Modeling Spatio-temporal Neighbourhood for Personalized Point-of-interest Recommendation**

**Next Point-of-Interest Recommendation with Inferring Multi-step Future Preferences**



# WWW 2022

 **STAM: A Spatiotemporal Aggregation Method for Graph Neural Network-based Recommendation**

 **Large-scale Personalized Video Game Recommendation via Social-aware Contextualized Graph Neural Network**



# RecSys 2022

**Global and Personalized Graphs for Heterogeneous Sequential Recommendation by Learning Behavior Transitions and User Intentions**

**TinyKG: Memory-Efficient Training Framework for Knowledge Graph Neural Recommender Systems**

**Modeling User Repeat Consumption Behavior for Online Novel Recommendation**

**Modeling Two-Way Selection Preference for Person-Job Fit**

