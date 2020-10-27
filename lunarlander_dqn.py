import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import gym 

# print(tf.__version__)
# print(gym.__version__)

env = gym.make('LunarLander-v2')
env.reset()
action_space = env.action_space
obs_space = env.observation_space

dqn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=obs_space.shape, activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_space.n, activation="linear")])

dqn_model_target = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=obs_space.shape, activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_space.n, activation="linear")])

dqn_model.summary()

lr = 0.001
episodes = 1000
timesteps = 10000
epsilon = 1
gamma = 0.99
N = 1000000
minibatch_size = 64
memory = deque(maxlen=N)
C=4
update_weights_every = 4
epsilon_decay = 0.995


dqn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=lr),
    loss='mse')

dqn_model_target.compile(
    optimizer=tf.optimizers.Adam(learning_rate=lr),
    loss='mse')

dqn_model_target.set_weights(dqn_model.get_weights())


total_rewards = []
episode_nums = []
print("Hyperparameters","\nGamma: ", gamma, "\nMemory: ", N, "\nminibatch size: ", minibatch_size,\
      "\nC: ", C, "\nupdate weights every: ", update_weights_every, "\nLearning rate: ", lr)
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(timesteps):
        # env.render()
        if random.random() <= epsilon:
            action = random.randrange(0,action_space.n)
        else:
            action = np.argmax(dqn_model.predict(np.array([state])), axis = 1)[0]
        
        next_state, reward, done, x = env.step(action)
        memory.append((state, action, reward, next_state, done))
        total_reward+=reward
        state = next_state

        if t%update_weights_every == 0: 
          if len(memory)>minibatch_size:
              minibatch = np.array(random.sample(memory, minibatch_size))
              states = np.array([transition[0] for transition in minibatch])
              actions = np.array([transition[1] for transition in minibatch])
              rewards = np.array([transition[2] for transition in minibatch])
              next_states = np.array([transition[3] for transition in minibatch])
              dones = np.array([transition[4] for transition in minibatch])

              y = rewards + gamma*np.max(dqn_model_target.predict(np.array(next_states)), axis=1)*(1-dones)
              targets = dqn_model_target.predict(np.array(states))
              targets[[j for j in range(minibatch_size)], actions] = y
              
              dqn_model.fit(states, np.array(targets), epochs=1, verbose=0)

        if t%C == 0:
          dqn_model_target.set_weights(dqn_model.get_weights())

        if done:
          break

    epsilon *= epsilon_decay
    if epsilon < 0.01:
        epsilon = 0.01
    total_rewards.append(total_reward)
    episode_nums.append(episode)
    print("episode: {}/{} -- reward: {}, epsilon: {}".format(episode+1, episodes, total_reward, epsilon))

    if (episode+1)%250 == 0 and episode > 0:
      plt.plot(episode_nums, total_rewards)
      plt.ylabel("reward")
      plt.xlabel("episode number")
      plt.show()
env.close()