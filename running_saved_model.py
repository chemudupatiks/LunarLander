# running_saved_model.py 
# Creates all the plots.

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import tensorflow as tf
import gym 

# Make Lunar lander env
env = gym.make('LunarLander-v2')
# Monitor each episode and save the videos to a directory
env = gym.wrappers.Monitor(env, "./trained_model_video", force=True,\
                           video_callable=lambda episode_id: True)
# Reset environment and get action and observation space
env.reset()
action_space = env.action_space
obs_space = env.observation_space

# Define model architecture and set hyperparameters 
dqn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=obs_space.shape, activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(action_space.n, activation="linear")])

dqn_model.summary()

# Load weights of the trained model which was saved
dqn_model.load_weights("run6/mymodel64x64")

# Make the model play the game for 5 episodes and record the weights
episodes = 5
timesteps = 5000 
total_rewards = []
episode_nums = []
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(timesteps):
        env.render()
        # Get action from the model for a given state
        action = np.argmax(dqn_model.predict(np.array([state])), axis = 1)[0]
        # Step through the env using the action
        next_state, reward, done, x = env.step(action)
        total_reward+=reward
        state = next_state
        if done:
          break

    total_rewards.append(total_reward)
    episode_nums.append(episode)
    print("episode: {}/{} -- reward: {}".format(episode+1, episodes, total_reward))
env.close()

# Make the reward vs episode_number plot
plt.plot(total_rewards, episode_nums)
plt.ylabel("reward")
plt.xlabel("episode number")
plt.show()