#based on stable baselines implementation https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

import gym
import tensorflow as tf
import os
import datetime
import stable_baselines3
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from stable_baselines3 import DDPG
from gym.envs.registration import registry, register, make, spec
from stable_baselines3.common.vec_env import SubprocVecEnv

tensorboard = TensorBoard(log_dir='logs')
 #Register the RockerLander environment
register(
    id='RocketLander-v0',
    entry_point='rocket_lander:RocketLander',#gym.envs.box2d:
    max_episode_steps=1000,
    reward_threshold=0,
)

timestep = 20000000
ENV  = 'RocketLander-v0'
timestamp = datetime.datetime.now()
filename = "ppo2_{}_{}_{}".format(ENV,timestep,str(timestamp)[:19])
path = '{}_tensorboard'.format(ENV[:-3])
env = gym.make('RocketLander-v0')
def make_env():
    return gym.make('RocketLander-v0')
n_cpu = 8
env = SubprocVecEnv([make_env for i in range(n_cpu)])

config = tf.ConfigProto()

with tf.Session(config=config):
  
  model = DDPG("MlpPolicy", env,n_steps=1024,nminibatches=256,lam=0.95,gamma=0.99,noptepochs=3,ent_coef=0.01,learning_rate=lambda _: 1e-4,cliprange=lambda _: 0.2, tensorboard_log=path,full_tensorboard_log=True,verbose=2)
  model.learn(total_timesteps=timestep, log_interval=1000 , callback = tensorboard)#15M timesteps and overnight run on a Macbook worked fine(still can improve).
  
  model.save(filename)
  model.save('./model/'+filename)
  
  print('Model Saved')
