# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 23:03:31 2020

@author: ckris
"""


import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import tensorflow as tf
import gym 

env = gym.make('LunarLander-v2')
env = gym.wrappers.Monitor(env, "./trained_model_video", force=True,\
                           video_callable=lambda episode_id: True)
env.reset()
action_space = env.action_space
obs_space = env.observation_space

dqn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=obs_space.shape, activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_space.n, activation="linear")])

dqn_model.summary()

dqn_model.load_weights("run6/mymodel64x64")

episodes = 5
timesteps = 5000
total_rewards = []
episode_nums = []
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(timesteps):
        env.render()
        action = np.argmax(dqn_model.predict(np.array([state])), axis = 1)[0]
        
        next_state, reward, done, x = env.step(action)
        total_reward+=reward
        state = next_state
        if done:
          break

    total_rewards.append(total_reward)
    episode_nums.append(episode)
    print("episode: {}/{} -- reward: {}".format(episode+1, episodes, total_reward))
env.close()