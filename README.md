# Learning Notes for Robot Learning(Part 2)
<br>**Skip back to [Part 1](https://github.com/NZ-Liam-Zhong/Awesome_AI_for_Robotics_Learning_Notes/blob/main/README.md)**<br>
**Skip to [Part 3](https://github.com/NZ-Liam-Zhong/Awesome_AI_for_Robotics_Learning_Notes_3)**<br><br>
As ChatGPT and RT2 appears, many researchers start to do reseaches about AI for robotics. Many scholars have different names for it, including embodied AI, smart robotics, etc. I will update the papers and content that I have read. Unlike the others, I won't just use python scirpt to get the information and abstract, I will share my thoughts after the reading. I promise that very single content listed below I have read it at least once. <br><br> This is **Part 2**. We divide them into many parts because it's hard to read the content if there are too many things in a single repo.<br><br>
![image](https://github.com/user-attachments/assets/2cdb6153-9914-4397-83ed-74157b8883e3)
(Image by Grok)

## 🤔Main Challenges🤔
1. Accuracy (Algorithm) 
2. Generality (Multi-task)
3. Inference speed 
4. Datasets
5. Sim2Real

## Tutorials & Slides
1.[Reinforcement Learning Basics by Fei-Fei Li](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf)
Slides for understanding basic concepts for reinforcement learning. **Institution： Stanford University** They have replicated the success of GPT o1, which is inspiring for robotics.<br>
2.[What can RL do 1?](https://neptune.ai/blog/reinforcement-learning-applications)<br>
[what can RL do2?](https://onlinedegrees.scu.edu/media/blog/9-examples-of-reinforcement-learning)<br>
3.[Lessons on Inverse RL](https://www.youtube.com/watch?v=qo355ALvLRI)<br>
demo agent != perfect agent<br>
![aaaa80452904669440ff7481ce201b0](https://github.com/user-attachments/assets/d4371879-43fb-4d90-beb9-d943855510cf)<br>
problem: policy and reward can have many solutions.<br>
![aaaa80452904669440ff7481ce201b0](https://github.com/user-attachments/assets/806693d3-452f-4007-9e5f-3080ddb9afa9)<br>
![54dc7c50d1d679e35f5bcf14e4e20ed](https://github.com/user-attachments/assets/37bbd003-492d-4edb-89bc-8d9078c96e79)<br>
4.Generative Adversarial Imitation learning uses GAN-related theory into imitation learning<br>
5.[GAIL paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf)GPT conclude:<br>
(1)Handling Compounding Error:<br>
Behavioral Cloning (BC) directly learns a policy using supervised learning on state-action pairs from expert demonstrations. However, BC is highly susceptible to compounding errors due to covariate shift—where the learner's policy diverges from the expert's policy during training, causing the model to perform poorly with small or noisy datasets.
GAIL, on the other hand, uses a generative adversarial approach to match the learner's occupancy measure (the distribution of state-action pairs) to the expert's, which does not suffer as much from compounding errors because it operates on entire trajectories and focuses on ensuring the learned policy mimics expert behavior more holistically.
More Robust with Less Data:<br>
BC often requires large amounts of expert data to perform well, as it treats imitation learning as a supervised task. In contrast, GAIL is more data-efficient because it learns by minimizing a Jensen-Shannon divergence between the learner's and expert's state-action distributions, leveraging a discriminator network to guide the policy improvement.
Better for Complex Tasks:<br>
GAIL can be more effective for complex, high-dimensional tasks (e.g., physics-based control tasks like humanoid locomotion) where BC may struggle due to the lack of sufficient expert data. GAIL leverages adversarial training to improve the policy without needing an explicit cost function, making it more suitable for high-dimensional environments.<br>
(2)Avoiding Reinforcement Learning in the Inner Loop:<br>
Traditional Inverse Reinforcement Learning (IRL) often requires reinforcement learning in an inner loop to optimize a learned cost function, making it computationally expensive and slow. GAIL simplifies this by directly learning a policy, bypassing the need for the IRL intermediate step.<br>
(3)Generative Approach:<br>
GAIL is closely related to Generative Adversarial Networks (GANs), where the learner's policy is trained to "fool" the discriminator into thinking the generated state-action pairs are from the expert. This allows GAIL to exactly mimic the expert's behavior, whereas BC may not achieve such precision without extensive fine-tuning.<br>
5.**Notes**<br>
1.Behaviour CLoning has severe Covariate shift problem, GAIL can improve that (error in new input)<br>
2.Traditional Inverse Reinforcement Learning requires reinforcement learning in an inner loop to optimize a learned cost function, making it computationally expensive and slow.<br>
6.[Stable-BC: Controlling Covariate Shift with Stable Behavior Cloning](https://arxiv.org/pdf/2408.06246)<br>
(1)**Covariate Shift**:Behavior cloning is a common imitation learning
 paradigm. Under behavior cloning the robot collects expert
 demonstrations, and then trains a policy to match the actions
 taken by the expert. This works well when the robot learner
 visits states where the expert has already demonstrated the
 correct action; but inevitably the robot will also encounter new
 states outside of its training dataset. If the robot learner takes
 the wrong action at these new states it could move farther from
 the training data, which in turn leads to increasingly incorrect
 actions and compounding errors. Existing works try to address
 this fundamental challenge by augmenting or enhancing the
 training data. By contrast, in our paper we develop the control
 theoretic properties of behavior cloned policies. Specifically, we
 consider the error dynamics between the system’s current state
 and the states in the expert dataset. From the error dynamics
 we derive model-based and model-free conditions for stability:
 under these conditions the robot shapes its policy so that its
 current behavior converges towards example behaviors in the
 expert dataset. <br>
 (2)Uses control theory methods. I will share this after I am familiar with control theory.
<br>
7.[OGBench](https://arxiv.org/pdf/2410.20092) A very good benchmark for goal conditioned RL.Offline goal-conditioned reinforcement learning (GCRL) is a major problem in re
inforcement learning (RL) because it provides a simple, unsupervised, and domain
agnostic way to acquire diverse behaviors and representations from unlabeled data
 without rewards. Despite the importance of this setting, we lack a standard bench
mark that can systematically evaluate the capabilities of offline GCRL algorithms.
 In this work, we propose OGBench, a new, high-quality benchmark for algorithms
 research in offline goal-conditioned RL. OGBench consists of 8 types of envi
ronments, 85 datasets, and reference implementations of 6 representative offline
 GCRLalgorithms. We have designed these challenging and realistic environments
 and datasets to directly probe different capabilities of algorithms, such as stitch
ing, long-horizon reasoning, and the ability to handle high-dimensional inputs and
 stochasticity. While representative algorithms may rank similarly on prior bench
marks, our experiments reveal stark strengths and weaknesses in these different
 capabilities, providing a strong foundation for building new algorithms.
 Project page: https://seohong.me/projects/ogbench
 Repository: https://github.com/seohongpark/ogbench

 7.[Implicit Q-Learning](https://www.cs.utexas.edu/~yukez/cs391r_fall2023/slides/pre_10-05_Marlan.pdf)<br>
 ![image](https://github.com/user-attachments/assets/a7fbff07-af7d-4099-a7a4-fe88c79ebde1)<br>
 ![image](https://github.com/user-attachments/assets/d373cb07-e867-4124-ac78-3c49413137db)<br>
 Policy extraction performed by advantage weighted regression<br>
 ![image](https://github.com/user-attachments/assets/4cf05b3b-2cfd-49cc-97f2-d26a78a6a2db)<br>
**Problem**: Assuming not a stachastic environment<br>
![image](https://github.com/user-attachments/assets/6a45e565-9dc5-480e-b250-ac41c78321a3)<br>
how to solve the problem?<br>
![image](https://github.com/user-attachments/assets/48986d50-149a-4619-ab3a-b83b35998ff2)<br>
![image](https://github.com/user-attachments/assets/2ecc42b7-d011-4f6a-9077-b64e977d0cc3)
error bound<br>
![image](https://github.com/user-attachments/assets/c89fa303-557e-49c0-a801-0ba612c33db1)


8.lambda big: optimistic small: pessimistic<br>

9.<br>
![image](https://github.com/user-attachments/assets/34c3dcf0-a791-4c67-87d3-92450cedf280)<br>

10.experiences about RL:[Value-Based Deep RL Scales Predictably](https://arxiv.org/pdf/2502.04327?)

11.[Scaling Test-Time Compute Without Verification
 or RL is Suboptima](https://arxiv.org/pdf/2502.12118)<br>
 ![image](https://github.com/user-attachments/assets/8c4a9279-f2ef-4cc0-be09-231af3d89ded)<br>


12.[Unsupervised-to-Online Reinforcement Learning](https://arxiv.org/pdf/2408.14785) sergey, seohong park<br>
![image](https://github.com/user-attachments/assets/df29bdc7-ac3b-43d6-9d0f-6e70a9b652c3)<br>
![image](https://github.com/user-attachments/assets/e0596b5e-8ca5-4d62-8692-13fa66ebe52b)<br>

13.[Efficient Online Reinforcement Learning
 Fine-Tuning Need Not Retain Offline Data](https://arxiv.org/pdf/2412.07762)<br>
 ![image](https://github.com/user-attachments/assets/08bcfc3c-76df-422e-8a84-8499ba91d763)<br>
 ![image](https://github.com/user-attachments/assets/41afd1ed-1350-4087-af16-2ba82dd179b2)<br>
![image](https://github.com/user-attachments/assets/e0c0a4b0-4a7a-40f6-826f-bd63831181b8)<br>

14.[GRPO explanation blog in Mandarin](https://zhuanlan.zhihu.com/p/21046265072)<br>

15.A(s, a) = Q(s, a) – V(s)<br>
(a)Generalized Advantage Estimation (GAE) <br>
![image](https://github.com/user-attachments/assets/31d2a65e-9db2-4413-a474-451685c6836c)<br>
(B)reward-weighted-regression<br>
![image](https://github.com/user-attachments/assets/ced52b3a-68aa-4b3c-81f5-1ea4272ff9c6)<br>
(c)[ADVANTAGE-WEIGHTED REGRESSION (AWR)](https://arxiv.org/pdf/1910.00177) AWR效果好于RWR<br>
![image](https://github.com/user-attachments/assets/27b9c2c3-145f-4638-a352-b0ee971d1a45)<br>

16.[Randomn Doubt is important in doing research,example"Is Noise Conditioning Necessary for Denoising Generative Models?"](https://arxiv.org/pdf/2502.13129)

17.[Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/e7663e974c4ee7a2b475a4775201ce1f-Paper-Conference.pdf)<br>
Contrastive Learning can be interpreted as actor critic method. High-dimentional method have advantages than other methods.<br>
(1)Method One<br>
![image](https://github.com/user-attachments/assets/ae796fbd-8d22-4564-97f5-c2b6cad48775)<br>
(2)Method two<br>
![image](https://github.com/user-attachments/assets/a60e42c0-64c9-40e4-a48e-e4b58b8f30b9)<br>
plus temperature<br>
![image](https://github.com/user-attachments/assets/9996b11a-77cb-409f-8e16-dc25b5f95d65)<br>

18.**Reward Shaping** in Goal Conditional RL<br>
1.[A good slide for reward shaping](https://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)<br>
![image](https://github.com/user-attachments/assets/f3abf176-5b67-4b72-9bc3-63c8af7d917c)<br>
Problem of not 0<br>
![image](https://github.com/user-attachments/assets/dff6b6d2-d23e-4597-95c5-dac68daf285b)<br>
keep iterating<br>
![image](https://github.com/user-attachments/assets/b3b6f670-4a68-4fe5-9fae-2b79b6d28215)<br>

19.[Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476)<br>
![image](https://github.com/user-attachments/assets/b9919a51-d073-4561-910e-3e10178bdd9f)<br>
Max-Entropy regularization to avoid overfitting<br>

20.[A Very Impressive paper Worth Reading,Foundation Policies with Hilbert Representations](https://arxiv.org/pdf/2402.15567)<br>
![image](https://github.com/user-attachments/assets/9d25f218-599f-4abe-bf5b-a7207d1555aa)<br>
![image](https://github.com/user-attachments/assets/403a9c9c-116f-41b7-a9dc-e8ebcf2ad314)<br>

21.[Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning](https://proceedings.mlr.press/v202/wang23al/wang23al.pdf)<br>
Quasimetric RL<br>
![image](https://github.com/user-attachments/assets/8d3c7ec1-c4fb-4277-82fc-0d9c1c6116a6)<br>

22.[Prof. Shiyu Zhao course on RL](https://www.shiyuzhao.net/opencourse)<br>
![image](https://github.com/user-attachments/assets/4fe036ef-af29-4a58-9e08-f1eb7e2bcf80)<br>

23. A = Q - V (Because only Q has a larger variance)<br>

24.on-policy -> off-policy : Actor-Critic <br>

25.[4D Reconstruction from a single video](https://arxiv.org/pdf/2407.13764)<br>
![image](https://github.com/user-attachments/assets/353b7929-6e42-4ae4-8b8f-afad62f1208d)<br>
Use multiple rules to help to optimize<br>

26.[Yann LeCun's world model for robot learning,Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models](https://latent-planning.github.io/static/paper.pdf)<br>
![image](https://github.com/user-attachments/assets/8c32644c-84be-4c76-8f4c-661e340447f2)<br>

27.[Triangular Inequality in Reinforcement Learning,An Inductive Bias for Distances: Neural Nets That Respect the Triangle Inequality](https://openreview.net/pdf?id=HJeiDpVFPr)<br>

28.[Book: The Mathematics of Reinforcement Learning](https://www.wim.uni-mannheim.de/media/Lehrstuehle/wim/doering/RL/RL_VORLESUNG.pdf)<br>
![image](https://github.com/user-attachments/assets/a797d956-85b3-4e88-b604-7711a2ac259e)<br>

29.In Goal Conditional RL, when reward set to minus, for example (-1,0), it will be quasi-metric.<br>
![34940efc44f0c28638a37b3c9a0514a](https://github.com/user-attachments/assets/e0df3a29-f963-4546-8cdc-5e869ceaf95e)<br>

30.[LIPSCHITZ-CONSTRAINED UNSUPERVISED SKILL DISCOVERY](https://arxiv.org/pdf/2202.00914)<br>
![image](https://github.com/user-attachments/assets/49b7ac5f-1b6e-4b84-9f3a-f1370801236f)<br>
 Lipschitz-constrained Skill Discovery (LSD)<br>

31.[GENSIM: GENERATING ROBOTIC SIMULATION TASKS VIA LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.01361)<br>
![image](https://github.com/user-attachments/assets/dc4b046e-d470-4fda-8ba4-8f91a0e43187)<br>

32.[GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs](https://arxiv.org/pdf/2410.03645)<br>
![image](https://github.com/user-attachments/assets/1545cfc0-3852-4ddb-ab52-56c33e58c351)<br>

33.[Appendix on Quasimetric space and optimal transport](https://openreview.net/references/attachment?id=hB5FFsuU3&name=supplementary_material)<br>

34.[OTTER:AVision-Language-Action Model with Text-Aware Visual Feature Extraction](https://arxiv.org/pdf/2503.03734)<br>
![image](https://github.com/user-attachments/assets/4e14b82d-8502-479d-86b0-9615c4c891e0)<br>
![image](https://github.com/user-attachments/assets/43351bf2-bdad-4063-ac30-57357d4b63ed)<br>
[Twitter](https://x.com/fangchenliu_/status/1900623471920738800)<br>

35.[Curating Demonstrations using Online Experience](https://arxiv.org/pdf/2503.03707?)<br>
![image](https://github.com/user-attachments/assets/0847ff93-9eb6-4768-a21e-618a87db2b96)<br>

36.[MAXINFORL: BOOSTING EXPLORATION IN REINFORCEMENT LEARNING THROUGH INFORMATION GAIN MAXIMIZATION](https://arxiv.org/pdf/2412.12098)<br>
The while paper is worth reading<br>
![image](https://github.com/user-attachments/assets/c2425d65-6fd1-483e-902a-728a49952634)<br>

37.[AgiBot World Colosseo: Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems](https://agibot-world.com/blog/agibot_go1.pdf)<br>
Combinations of existing works<br>
The pictures are really good<br>
![image](https://github.com/user-attachments/assets/5924e581-bdcb-494a-9a71-b79be635f5ac)<br>

38.[Flow Q Learning](https://arxiv.org/pdf/2502.02538)<br>
![c8ece3557abc24a5f636d5a15ca1afa](https://github.com/user-attachments/assets/1657809d-87a3-41a4-9935-c3eb0bfc2796)<br>
![b118165055093bcb5f40000c71f3087](https://github.com/user-attachments/assets/4af0433a-5019-46b3-ad5a-cda3c9a081f9)<br>

39.[Tutorials on Diffusion Model](https://www.eecs.umich.edu/courses/eecs442-ahowens/fa23/slides/lec11-diffusion.pdf)<br>

40.[Nvidia GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)<br>
![image](https://github.com/user-attachments/assets/acae210f-9151-4414-a4f0-ee05308ceb06)<br>
![image](https://github.com/user-attachments/assets/5791013c-817b-47ed-a47c-9484c33fc955)<br>
vlm+skill token+dp<br>

41.[Gemini Robotics: Bringing AI into the Physical World](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf)<br>
![image](https://github.com/user-attachments/assets/3b90f2ec-9fe4-4324-991e-a138e2339652)<br>

42.[Slides about Multi-Agent RL](https://cs.uwaterloo.ca/~ppoupart/teaching/cs486-spring23/slides/cs486-lecture23-annotated.pdf)<br>
Uinversity of Toronto<br>

43.[Tutorials on MARL](https://wnzhang.net/tutorials/marl2018/index.html)<br>
UCL<br>

44.[Tutorials on DDPG & MADDPG](https://medium.com/machine-intelligence-and-deep-learning-lab/a-tutorial-on-maddpg-53241ae8aac)<br>
![image](https://github.com/user-attachments/assets/b887685c-048d-4cd4-8ac4-7d95a3db3687)<br>
![image](https://github.com/user-attachments/assets/1e88b63a-977e-49f7-aa62-c37f449d5119)<br>

45.[Tutorials on offline multi-agent reinforcement learning](https://icml.cc/media/icml-2022/Slides/17592.pdf)<br>




## Industry
Here we update some tech advance in the industry🔥🔥🔥
There are too many start-ups doing the same thing, boring and useless. So, we will only update the AI for robotic start-ups who make **innovative products**

## 🖊Notes🖊



