#based on stable baselines implementation https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from gym.envs.registration import registry, register, make, spec

#registering custom environment
register(
    id='RocketLander-v0',
    entry_point='rocketlander.rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
)

# Create and wrap the environment
env = gym.make('RocketLander-v0')
env = gym.wrappers.Monitor(env, "./video", force=True)
env = DummyVecEnv([lambda: env])



# Load the trained agent
#ppo2_RocketLander-v0_20000000_2019-05-05 03/26/38.pkl
model = PPO2.load("./model/ppo2_RocketLander_15000000.pkl")


#Trained agent in action
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
    	break
