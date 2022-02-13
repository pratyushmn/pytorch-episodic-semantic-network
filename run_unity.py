from mlagents_envs.base_env import DecisionStep
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from random import seed
from random import randint
from main.agent import Agent

def trainAgent(agent, env, num_episodes):   
    for episode in range(num_episodes):
        print('Finding reward # ', episode + 1, end='\r')
        done = False
        state = env.reset()
        time = 0

        while not done:
            action = agent.act(state, time)
            state, reward, done, info = env.step(action)
            # print("State: {}, Reward: {}, Done: {}, Info:{}, Action: {}".format(state, reward, done, info, action))
            agent.learn(state, reward, 0)
            time += 1

        print('\nFound reward: ', episode + 1)
        print('Time to reward: ', time)
        agent.policyN= 1


if __name__ == "__main__":
    envU = UnityEnvironment()
    env = UnityToGymWrapper(envU, allow_multiple_obs=False)
    agent = Agent.StandardAgent()
    print("Action Space Size: {}".format(env.action_space.n))
    print("Observation Space Shape: {}".format(env.observation_space.shape))

    num_episodes = 1000

    trainAgent(agent, env, num_episodes)