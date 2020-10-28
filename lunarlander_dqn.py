# lunarlander_dqn.py
# Design and train the model and finally save its weights 

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

import tensorflow as tf
import gym 

# Make sure TensorFlow and Gym libraries are properly installed
# print(tf.__version__)
# print(gym.__version__)


# Initialize the Lunar lander env 
env = gym.make('LunarLander-v2')
env.reset()
action_space = env.action_space
obs_space = env.observation_space

# Define model architecture 
dqn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=obs_space.shape, activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_space.n, activation="linear")])

# Define target model architecture (same as the main model)
dqn_model_target = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=obs_space.shape, activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_space.n, activation="linear")])

dqn_model.summary()

# Set hyperparameters
lr = 0.001
episodes = 1000
timesteps = 10000
epsilon = 1
gamma = 0.99
N = 1000000 # memory size
minibatch_size = 64
memory = deque(maxlen=N)
C=4 # number of timesteps before the target model weights are updated to main model weights
update_weights_every = 4 # number of timesteps before the main model weights are updated 
epsilon_decay = 0.995

# Initialize the models with random weights
dqn_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=lr),
    loss='mse')

dqn_model_target.compile(
    optimizer=tf.optimizers.Adam(learning_rate=lr),
    loss='mse')

# Set the target model weights to the main model weights
dqn_model_target.set_weights(dqn_model.get_weights())


total_rewards = []
episode_nums = []
print("Hyperparameters","\nGamma: ", gamma, "\nMemory: ", N, "\nminibatch size: ", minibatch_size,\
      "\nC: ", C, "\nupdate weights every: ", update_weights_every, "\nLearning rate: ", lr)
    
# DQN model with replay experience
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(timesteps):
        # env.render()
        
        # Get an random action with probability epsilon
        if random.random() <= epsilon:
            action = random.randrange(0,action_space.n)
        else:
            action = np.argmax(dqn_model.predict(np.array([state])), axis = 1)[0]
        
        # Step through the env and save to memory
        next_state, reward, done, x = env.step(action)
        memory.append((state, action, reward, next_state, done))
        total_reward+=reward
        state = next_state

        # Update main model weights
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
        
        # Update target model weights
        if t%C == 0:
          dqn_model_target.set_weights(dqn_model.get_weights())

        if done:
          break

    # Update epsilon 
    epsilon *= epsilon_decay
    if epsilon < 0.01:
        epsilon = 0.01
    total_rewards.append(total_reward)
    episode_nums.append(episode)
    print("episode: {}/{} -- reward: {}, epsilon: {}".format(episode+1, episodes, total_reward, epsilon))

    # Plot reward vs epsiode every 250 episodes
    if (episode+1)%250 == 0 and episode > 0:
      plt.plot(episode_nums, total_rewards)
      plt.ylabel("reward")
      plt.xlabel("episode number")
      plt.show()
      
env.close()
# Save the main model's weights to a file
dqn_model.save_weights("./mymodel64x64")