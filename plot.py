import numpy as np
import matplotlib.pyplot as plt


# file_path = 'runs/PK/Consolidation/'

# for i in range (1, 3):
#     cumulative_reward = np.load(file_path + '{}/cumulative_reward.npy'.format(i))

#     plt.title('Cumulative Reward')
#     plt.plot(cumulative_reward)
#     plt.show()

#     reward_rate = np.load(file_path + '{}/reward_rate.npy'.format(i))

#     plt.title('Reward Rate')
#     plt.plot(reward_rate)
#     plt.show()

cumulative_reward = np.load('data/cumulative_reward.npy')

plt.title('Cumulative Reward')
plt.plot(cumulative_reward)
plt.show()

finding_time = np.load('data/finding_time.npy')

plt.plot(finding_time)
plt.show()

reward_rate = np.load('data/reward_rate.npy')

plt.title('Reward Rate')
plt.plot(reward_rate)
plt.show()

cumulative_reward = np.load('data/cumulative_reward_per_episode.npy')

plt.title('Cumulative Reward')
plt.plot(cumulative_reward)
plt.show()

reward_rate = np.load('data/reward_rate_per_episode.npy')

plt.title('Reward Rate')
plt.plot(reward_rate)
plt.show()