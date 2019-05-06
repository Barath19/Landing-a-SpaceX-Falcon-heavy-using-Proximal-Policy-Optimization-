#based on stable baselines implementation https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

import gym
import tensorflow as tf
import os
import datetime
import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from gym.envs.registration import registry, register, make, spec

 #Register the RockerLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
)

#Defining utils
n_cpu = 4
timestep = 20000000
ENV  = 'RocketLander-v0'
timestamp = datetime.datetime.now()
filename = "ppo2_{}_{}_{}".format(ENV,timestep,str(timestamp)[:19])

# Create log dir
path = '{}_tensorboard'.format(ENV[:-3])
#os.makedirs(path, exist_ok=True)
#os.makedirs("Monitor_Log", exist_ok=True)

env = gym.make(ENV)#Creating the Environment
#env = gym.wrappers.Monitor(env, "./video", force=True)
env = Monitor(env, 'Monitor_Log', allow_early_resets=True)
#env = DummyVecEnv([lambda: env])
env = SubprocVecEnv([lambda: gym.make('RocketLander-v0') for i in range(n_cpu)])

config = tf.ConfigProto()
#if GPU uncomment below couple of lines of code
#config = tf.ConfigProto(device_count = {'GPU': 0})
#config.gpu_options.allow_growth = True

#Let's run a tensorflow session 
with tf.Session(config=config):
  
  model = stable_baselines.ppo2.PPO2(MlpPolicy, env,n_steps=1024,nminibatches=256,lam=0.95,gamma=0.99,noptepochs=3,ent_coef=0.01,learning_rate=lambda _: 1e-4,cliprange=lambda _: 0.2, tensorboard_log=path,full_tensorboard_log=True,verbose=2)
  model.learn(total_timesteps=timestep, log_interval=1000)#15M timesteps and overnight run on a Macbook worked fine(still can improve).
  
  model.save(filename)
  model.save('./model/'+filename)
  
  print('Model Saved')
