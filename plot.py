import numpy as np
import matplotlib.pyplot as plt

cumulative_reward = np.load('data/cumulative_reward.npy')

plt.plot(cumulative_reward)
plt.show()

finding_time = np.load('data/finding_time.npy')

plt.plot(finding_time)
plt.show()

reward_rate = np.load('data/reward_rate.npy')

plt.plot(reward_rate)
plt.show()