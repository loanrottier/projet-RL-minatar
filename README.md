# Project Reinforcement Learning: MinAtar games
Reinforcement Learning: Solving MinAtar games using Deep Q-Learning

Overview:
1) Understanding Deep Q-Learning
2) Solving games with _gymnasium_
3) Solving MinAtar games

## Understanding Deep Q-Learning
Source: https://huggingface.co/learn/deep-rl-course/unit0/introduction, units 1 to 4

The main objective is to create an agent able to take an action at each state. To do this, we use reinforcement learning. Using this allows the agent to learn from the previous episodes to take the action that will lead to a higher reward. The objective is to find the optimal policy. 

There are two types of policies:
* policy-based method: we play an entire episode to associate a cumulative reward with this episode and then this sequence of actions
* value-based method: we associate with each action and each state a reward value. While trained, the agent will choose the action with the higher value, and then the higher cumulative reward (in this case, we choose an Epsilon-Greedy policy). In this case, the policy is the choice of the action given the values.
In this project, we will use policy-based methods.

For the Atari games, sometimes, we need some preprocessing to keep only the important information. We can reduce the state space and grayscale it if the colors are not important. We can also crop part of the screen for some games that can have borders or surroundings.
For some games, it is also important to stack 4 frames to tackle the temporal limitations, because we sometimes need to know the previous state, to know the direction of the objects for example. 

#### Policy gradient with Pytorch (unit 4)

The objective is to find the optimal policy that will maximize the expected cumulative reward. We have the following policy function $\pi_\theta(s) = \mathbb{P}[A | s; \theta]$ and the objective function $J(\theta)$ that computes the expected cumulative reward. Thanks to the gradient ascent, we look for the $\theta$ that maximizes J. The gradietn ascent update the $\theta$ in the following way: $ \theta \leftarrow \theta + \alpha * \nabla_\theta J(\theta)$. In reality, we estimate the gradient because we do not have all the probabilities to compute the true gradient. We use the Policy Gradient theorem that reformulates the objective function $J$ to made it differentiable: $\nabla_\theta J(\theta) = E_{\pi_\theta} (\nabla_\theta log \pi_\theta(a_t|s_t) R(\tau))$

We let the agent interact during all the episodes. If the agent win the game, all the probability of the taken actions are increased. If it loses, the probability decrease.

## Solving  games  with _gymnasium_

Source: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Following the previous tutorial, we construct an DQN agent able to play to different gymnasium games. 

The first one is CartPole. The aim is to maintain the vertical bar the longer to have the most reward. The episode ends if it last more than 500 or if the bar falls or if the cart move too far away from the center (+/- 2.4 units)

The following plot shows the duration of each episode. We can see that after 180 episodes the agent knows how to play and reach the maximum duration for almost all episodes. 



The following videos show the evolution of the agent along the episodes. 
https://github.com/user-attachments/assets/62b17825-d776-4b1d-9a26-b2dec3aefc62

https://github.com/user-attachments/assets/7656aacf-f94c-48de-b512-5e8ecfad8c83

https://github.com/user-attachments/assets/e1bda450-4eab-46bf-a45a-8515a655739a

https://github.com/user-attachments/assets/51efc182-4577-4eed-b586-5c077dcbf662

https://github.com/user-attachments/assets/f98d5284-5f14-471d-b43f-eaef7710a09c









