# Landing-a-SpaceX-Falcon-heavy-using-Proximal-Policy-Optimization
While rocket landing problems are typically solved through conventional trajectory optimization techniques combined with heuristic control, recent developments in deep learning suggest that neural networks are able to approximate the Bellman equation and control spacecraft optimally in real-time using Reinforcement Learning. This project uses an RL agent to control the landing of a virtual rocket in a custom OpenAI gym environment.

Let us try to land a SpaceX Falcon Heavy Rocket in simulation using Reinforcement learning. Reinforcement learning is a technique that lets an agent learn how best to act in an environment using rewards as its signal. OpenAI released a library called Gym that lets us train AI agents really easily. We'll also use Stable Baselines and gym libraries to build an RL agent capable of landing a rocket perfectly. The specific algorithm we will be using is called proximal policy optimization, this is an improved version of actor-critic algorithm.

![alt-text](ezgif.com-video-to-gif.gif)

# Dependencies
pip install gym box2d-py stable_baselines

# Usage

python ppo_RL_train.py (To train the agent)

python ppo_RL_pretrained.py (To run a pretrained agent)
