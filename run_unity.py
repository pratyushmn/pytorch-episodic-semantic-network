from mlagents_envs.base_env import DecisionStep
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from main.agent import Agent
import numpy as np
import time as t
import argparse

def trainAgent(agent, env, num_episodes, save_path = None):   
    reward_rate = []
    cumulative_reward = []
    finding_time = []
    total_reward = 0
    steps = 0

    for episode in range(num_episodes):
        print('Finding reward # ', episode + 1)
        done = False
        state = env.reset()
        prevState = state
        time = 0

        while not done:
            prevState = state
            action = agent.act(state, time)
            state, reward, done, info = env.step(action)

            # done = reward > 0
            # done from env step not working at all
            # reward at first step after env epoch reset (when unity automatically does it) goes to negative total reward
            # over last epoch (ie. delta of setting it back to 0)
            # print("Prev State: {}, State: {}, Reward: {}, Done: {}, Info:{}, Action: {}".format(prevState, state, reward, done, info, action))

            # if reward > 0:
            #     print("Reward > 0")
            #     print(done)
            #     t.sleep(5)

            agent.learn(state, reward, 0)

            time += 1
            steps += 1
            total_reward += reward
            reward_rate.append(total_reward/steps)
            cumulative_reward.append(total_reward)

        print('Found reward: ', episode + 1)
        print('Time to reward: ', time)
        finding_time.append(time)
        agent.policyN= 1

    # save data
    if save_path is not None:
        reward_rate = np.array(reward_rate)
        cumulative_reward =np.array(cumulative_reward)
        finding_time = np.array(finding_time)

        np.save(save_path + "reward_rate", reward_rate)
        np.save(save_path + "cumulative_reward", cumulative_reward)
        np.save(save_path + "finding_time", finding_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None, help="Input filepath for Unity Environment executable")
    parser.add_argument("-n", "--num", type=int, default=120, help="Number of episodes to run on the Unity Environment")
    parser.add_argument("-r", "--render", action='store_false', help="Toggle to not render the environment")
    parser.add_argument("-k", "--knowledge", action='store_false', help="Toggle to turn off prior knowledge")
    parser.add_argument("-f", "--file", type=str, default=None, help="Directory to save output to (only for test)")
    
    args = parser.parse_args()

    envU = UnityEnvironment(file_name=args.path, no_graphics=(not args.render))
    env = UnityToGymWrapper(envU, allow_multiple_obs=False)
    agent = Agent.StandardAgent(contextDimension=(env.observation_space.shape[0] - 2), actionSpace=env.action_space.n, priorKnowledge=args.knowledge)
    print("Action Space Size: {}".format(env.action_space.n))
    print("Observation Space Shape: {}".format(env.observation_space.shape))

    trainAgent(agent, env, args.num, args.file)