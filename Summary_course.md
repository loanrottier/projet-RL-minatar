This document explains the Deep Reinforcement Learning from the Hugging Face Course.

Source : https://huggingface.co/learn/deep-rl-course/unit1/rl-framework

# Unit 1 - Introduction to Deep Reinforcement Learning

State $S_0$ -> Action $A_0$ -> Reward $R_1$ -> Next State $S_1$

Agent's goal : maximize the expected return (= cumulative rewards)

RL process = Markov decision process (just takes the current state to take an action)

State : all the environment $\neq$ observation : part of the environment

Action space : all possible actions from the state

Reward : $R(\tau)$ = cumulative returns = all rewards from t = 0 to t = $\tau$

but can sum up like this, discount rate \gamma (between 0 and 1) because some rewards are less likely 

Large \gamma : agent cares more about long-term reward
    
$R(\tau) = sum \gamma^k r_{t+k+1}$

Type of tasks:
- episodic if begin and end 
- continuing (no terminal state)

Exploration/Exploitation trade-off
- Exploration: try random actions to find more information about the environment
- Exploitation: use known information to maximize the reward
If just exploitation: can fall into common trap

How to solve the RL problem ?
$\pi$ function (Policy): $\pi(state)$ = action
$\pi$ = function we want to learn -> find $\pi^*$ = optimal policy = maximize the reward
- Policy-Based methods : say to the agent which action to choose ; mapping between each state and the action to take or probability distribution to each action at this state
    - deterministic : always the same action to this state
    - stochastic : probability distribution over actions
- Value-Based methods : expected discounted return being at that state (expectation(sum of rewards by $gamma^t | S_t = s$))
    Chosse the action with the biggest Value

# Unit 2 - Introduction to Q-Learning

Obj: find the optimal policy
* policy-based methods: train directly the policy (= NN) ; no value function ; the training define the behavior of the policy
* value-based methods: train the value and then take an action given this value ; policy is defined by hand (eg: take the action with the biggest value) ; train the value function (=NN)

    Often: Espilon-Greedy policy, not max but close, allows to handles Exploitation-Exploration trade-off
    * state value function: expected return if the agent starts at state s : $v_{\pi}(s) = E_{\pi}(G_t|S_t=s)$
    * action value function: $Q_{\pi}(s,a) = E_{\pi}(G_t|S_t = s, A_t = a)$: takes an action-state pair

Simplification thanks to the Bellman equation: allows to not compute again all the values function at step $s+1$
$V_{\pi}(s) = E_{\pi}(R_{t+1} + \gamma * V(_{\pi}(S_{t+1}) | S_t  = s)$
The value of $\gamma$ impacts the weight put on the short-term reward (if $\gamma$ small) and on the next-state value (if $\gamma$ large)

Learning strategies:
* Monte-Carlo use an entire episode before Learning ; wait until the end of the episode then calculate $G_t$ and update $V_t$
$V(S_t) <- V(S_t) + \alpha (G_t - V(S_t))$
The agent will learn as more episodes are played
* Temporal Difference learn with only one step: update $V(S_t)$ at each step, we estimate $G_t$ by $R_{t+1} + \gamma * V(S_{t+1})$

Q-Learning : off-policy value-based methods with TD approach

off-policy: different policy for acting and training (contrary = on-policy, same policy)
    
train Q-function = action-value function
    
Q-table: each cell = one combination of action and state
    
The algorithm will update the table during the training, when it's done we have an optimal policy
    
Algorithm:

1. initialisation of the Q-table at 0
    
2. choose an action, epsilon-greedy policy ; proba $1 - \epsilon$ to do exploitation (action with the highest state-action value) ; proba $\epsilon$ to do Exploration (random action) ; at the beginning, large $\epsilon$ and reduce as the steps go
    
3. take the action $A_t$, observe $R_{t+1}$ and $S_{t+1}$
    
4. update the Q-table
    
$Q(S_t, A_t) <- Q(S_t, A_t) + \alpha (R_{t+1} + \gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t))$

TD target : $R_{t+1} + \gamma * max_a Q(S_{t+1}, a)$
            
TD error: $R_{t+1} + \gamma * max_a (Q(S_{t+1}, a) - Q(S_t, A_t)$

# Unit 3 - Deep Q-learning with Atari games

Large state space, updating Q-table is inefficient

Deep Q-learning uses a NN that takes a state and approximates Q-values for each action

Atari games : 210x160 pixels, 3 colors (going from 0 to 255) --> 256^(210*160*6)

Deep Q-network : Convolutional Layers, Fully connected Layers, Q-value for each action then epsilon-greedy policy

Pre-processing:
* Reduce the state space to 84*84 and grayscale it (because the colors are not important)
* We can also crop part of the screen for some games
* Stack 4 frames to tackle the temporal limitations

Convolutional layers (3) ; Fully connected layers

Deep Q-learning: we do not update the Q values but we create a loss function that compares the Q-values prediction and the Q-target + gradient descent to update the weights our the Deep Q-Network to approximate the Q-value better

Q-loss: $y_i - Q(\phi_j, a_j, \theta) = [R_{t+1} + \gamma max_a Q(S_{t+1},a) - Q(S_t, A_t)]$
    
Two phases: 
1. sampling, do actions and store the experience tuples in a replay memory
2. training, select a small batch of tuples randomly and learn from this batch using a gradient descent udpate step
            
Details of the algorithm

Instability ? because non-linear Q-value function (NN) and bootstrapping
Solutions: 
* Experience replay (more efficient use of experiences) : not just learn from the experience but save them to use them again after (can learn from the same experience multiple times)
 + avoid forgetting previoys experiences and reduce the correlation between experiences (aka catastrophic forgetting, eg: forget about the actions of the first level while in the second one)
Solution: replay buffer (stores experience tuples while interacting with the environment and then sample a small batch if tuples, also learn from more previous experience) + random sampling 
* Fixed Q-target (stabilize the training) : difference between the TD target and the current Q-value
  Estimate the Q-target (but same parameters with the Q-estimation then both shift as we update the parameters)
  Solution: separate network with fixed parameters for estimating the TD Target (copy the parameters every C steps to update the target network)
* Double Deep Q-learning (handle the problem of the overestimation of Q-values) (by Hado van Hasselt), against the overestimation of Q-values (at the beginning of the training, we do not know the best action to take, the algorithm can do mistake and then lead to false positives)
* Solution:
    * 1 NN for the DQN network (select best action)
    * 1 NN for Target Network (calculate the target Q-value)

# Unit 4 - Policy gradient with PyTorch

Until now, value-based methods (policy exists because of the action value estimates, policy just a function)

Policy-based method, optimize the policy directly

Obj: find the optimal policy $\pi^*$ that will maximize the expected cumulative reward

Parametrize the policy function, eg: $\pi_\theta(s) = \mathbb{P}[A | s; \theta]$ -probability distribution over actions at that state-

Obj: maximize the performance of the parametrized policy using gradient ascent --> control the parameter \theta

Objective function $J(\theta)$ = expected cumulative reward ; look for \theta that maximizes J 

Policy-based vs policy-gradient method

Often, on-policy (update by our most recent version of \pi_\theta)
    * policy-based methods: search directly for the optimal policy ; optimizing of \theta indirectly by maximizing the local approximation of J (eg: hill climbing, simulated annealing, evolution strategies)
    * policy-gradient methods: search directly for the optimal policy; direct optimization by gradient ascent

Advantages of policy-gradient method:
    * simplicity of integration, estimate the policy without storing additional data
    * can learn a stochastic policy, don't need to implement an exploration/exploitation trade-off + no problem of perceptual aliasing (when two sates seem the same but need different actions)
        Same probability for both actions will allow to not always choose the same one
    * more efficient in high-dimensional action spaces and continuous actions spaces
    * better convergence properties (value-based method can change dramatically with an arbitrarily small change in the estimated action values); change smoothly over time

Disadvantages:
    * CV to local maximum instead of global one
    * slower, longer to train
    * high variance

Policy-gradient method
    
Parametrized stochastic policy
    
Goal: control the probability distribution of actions (good actions are sampled more frequently in the future)
    
Let the agent interact during an episode; if we win, all actions are noted as good and must be more sampled in the future
    
If we win, we increase the probability P(a|s) if the episode was winning and we decrease it if we lost
    
Details:

Stochastic policy: $\pi_{theta}(s) = P(A|s; \theta)$ (probability distribution over actions at that state)

Score $J(\theta)$ to measure the efficiency of the policy = objective function
    
= Performance of the agent given a trajectory (state action sequence) = expected cumulative reward
    
$J(\theta) = E_{\tau ~ \pi}(R(\tau))$
        
$E_{\tau ~ \pi}(R(\tau))$ = expected return = weighted average
     
$J(\theta) = \sum_\tau P(\tau;\theta)R(\tau)$
        
$R(\tau)$ = return from an arbitrary trajectory
            
$P(\tau;\theta)$ = probability of each possible trajectory = $\Pi_{t=0} P(s_{t+1}|s_t, a_t) \pi_\theta(a_t|s_t)$

Objective: $max_\theta J(\theta) = E_{\tau ~\pi_\theta}(R(_tau))$ --> weights that maximize the expected return

Methodology: gradient ascent: update step: $\theta \leftarrow \theta + \alpha * \nabla_\theta J(\theta)$

But we need the derivative of J --> the gradient is estimated (because we do not have all probabilities for each possible trajectory)
    
+ need to differentiate the state distribution (Markov Decision Process dynamics) --> Policy Gradient Theorem (reformulate the objective function into a differentiable function)

Policy gradient theorem: $\nabla_\theta J(\theta) = E_{\pi_\theta} (\nabla_\theta log \pi_\theta(a_t|s_t) R(\tau))$


Reinforce algorithm (Monte Carlo Reinforce): estimated return from an entire episode to update the policy parameter \theta 
   
Loop: policy $\pi_\theta$ to collect an episode $\tau$
    
use the episode to estimate the gradient $\hat\{g} = \nabla_\theta J(\theta) = \sum_{t=0} \nabla_\theta log \pi_\theta(a_t|s_t)R(\tau)$
        
with $\pi_\theta(a_t|s_t)$ = probability of the agent to select action at from state st given our policy
            
$\nabla_\theta log \pi_\theta(a_t|s_t)$ = direction of the steepest increase of the (log) probability of selecting action at from state st
            
$R(\tau)$ = cumulative return = scoring function (if high we push up the probabilities of the (state, action) combinations)
       
update the weights $\theta = \theta + \alpha \hat{g}$

Collect multiple episodes to estimate the gradient (1/m scaling factor)

