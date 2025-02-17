# Learning Notes for Robot Learning(Part 2)
<br>**Skip back to [Part 1](https://github.com/NZ-Liam-Zhong/Awesome_AI_for_Robotics_Learning_Notes/blob/main/README.md)**<br><br>
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
7.[OGBench](https://arxiv.org/pdf/2410.20092) A very good benchmark for goal conditioned RL.


## Industry
Here we update some tech advance in the industry🔥🔥🔥
There are too many start-ups doing the same thing, boring and useless. So, we will only update the AI for robotic start-ups who make **innovative products**

## 🖊Notes🖊



