# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:02:50 2020

@author: ckris
"""
import numpy as np 
import re
import matplotlib.pyplot as plt

def read_plain_numbers(filename):
    with open(filename) as f:
        lines = f.readlines()
        rewards = np.array(lines, dtype=float)
        return rewards
    
def read_output_txt(filename):
    with open(filename) as f:
        lines = f.readlines()
        rewards = []
        for line in lines:
            reward_match = re.findall( r'reward: (-?\d+\.\d+)', line)
            if len(reward_match)>0:
                rewards.append(reward_match[0])
        return np.array(rewards, dtype=float)
    
episodes = np.arange(750)+1

rewards_lr_0_0005 = read_output_txt("rewards_lr_0_0005.txt")
rewards_lr_0_001 = read_plain_numbers("rewards_lr_0_001")
rewards_lr_0_01 = read_plain_numbers("rewards_lr_0_01")

plt.plot(episodes, rewards_lr_0_0005, 'r', label="lr = 0.0005")
plt.plot(episodes, rewards_lr_0_001, 'g', label="lr = 0.001")
plt.plot(episodes, rewards_lr_0_01, 'b', label="lr = 0.01")
plt.ylim(-1000, 300)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.title("rewards with varying learning rates")
plt.show()



rewards_eps_decay_0_9 = read_output_txt("rewards_eps_decay_0_9.txt")
rewards_eps_decay_0_99 = read_output_txt("rewards_eps_decay_0_99.txt")
rewards_eps_decay_0_995 = rewards_lr_0_001

plt.plot(np.arange(rewards_eps_decay_0_9.size), rewards_eps_decay_0_9, 'r', label="eps_decay = 0.9")
plt.plot(np.arange(rewards_eps_decay_0_99.size), rewards_eps_decay_0_99, 'g', label="eps_decay = 0.99")
plt.plot(np.arange(rewards_eps_decay_0_995.size), rewards_eps_decay_0_995, 'b', label="eps_decay = 0.995")
# plt.ylim(-1000, 300)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.title("rewards with varying epsilon decay")
plt.show()


rewards_shallow_arch = read_plain_numbers("rewards_shallow_arch.txt")
rewards_deep_arch = rewards_lr_0_001

plt.plot(np.arange(rewards_shallow_arch.size), rewards_shallow_arch, 'r', label="shallow (1x128)")
plt.plot(np.arange(rewards_deep_arch.size), rewards_deep_arch, 'g', label="deep (2x64)")
# plt.ylim(-1000, 300)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.title("rewards with shallow architecture vs deep architecture")
plt.show()

rewards_C_1 = read_output_txt("rewards_C_1.txt")
rewards_C_4 = rewards_lr_0_001
rewards_C_16 = read_output_txt("rewards_C_16.txt")

plt.plot(np.arange(rewards_C_1.size), rewards_C_1, 'r', label="C = 1")
plt.plot(np.arange(rewards_C_4.size), rewards_C_4, 'g', label="C = 4")
plt.plot(np.arange(rewards_C_16.size), rewards_C_16, 'b', label="C = 16")
# plt.ylim(-1000, 300)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.title("rewards with varying C")
plt.show()

rewards_gamma_0_9 = read_output_txt("rewards_gamma_0_9.txt")
rewards_gamma_0_99 = rewards_lr_0_001
rewards_gamma_0_999 = read_plain_numbers("rewards_per_gamma_0_999.txt")

plt.plot(np.arange(rewards_gamma_0_9.size), rewards_gamma_0_9, 'r', label="gamma = 0.9")
plt.plot(np.arange(rewards_gamma_0_99.size), rewards_gamma_0_99, 'g', label="gamma = 0.99")
plt.plot(np.arange(rewards_gamma_0_999.size), rewards_gamma_0_999, 'b', label="gamma = 0.999")
plt.ylim(-2000, 300)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.title("rewards with varying gamma")
plt.show()


