import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from gym.envs.registration import registry, register, make, spec
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
model = PPO2.load("./model/ppo2_RocketLander_15000000.pkl")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
    	break