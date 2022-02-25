import numpy as np
import torch
from .components import EpisodicLearner
from .components import SemanticLearner
from .components import NavigationLearner
from ..lib.utils import activatePolicy

class StandardAgent():
    def __init__(self, episodicUnits = 980, contextDimension=0, actionSpace=8):
        self.learners = []
        self.learners.append(EpisodicLearner.EpisodicLearner(episodicUnits, numContext=contextDimension))
        self.learners.append(SemanticLearner.SemanticLearner(episodicUnits))
        self.navigation = NavigationLearner.NavigationLearner(episodicUnits, actionSpace=actionSpace)

        # Episodic, semantic, and overall
        self.goals = [np.zeros(episodicUnits), np.zeros(episodicUnits), 
                np.zeros(episodicUnits)]

        # Policy to switch between episodic and semantic network
        self.tau = 2000
        self.policyN = 1

        # last action
        self.action = 0

    def act(self, state, trialTime):
        # Episodic goal
        self.ca1Now = self.probeCA1(state)
        self.currState = state
        self.goals[0] = self.learners[0].forward(state)
        self.learners[0].predict_state_val(state)
        
        # Semantic goal
        self.goals[1] = self.learners[1].forward(self.ca1Now)

        # Overall goal
        self.goals[2] = (self.policyN * self.goals[0] + (1 - self.policyN) 
                * self.goals[1])

        # Choose an action based on goal and predictions from nav network
        self.action = self.navigation.choose(self.goals[2], self.ca1Now, trialTime)

        # Decay policy
        self.decayPolicy()

        return self.action

    def learn(self, state, reward, delay):
        # We need to take a step in the environment
        # before we learn so that the navigation net has
        # access to the next state given the action taken
        self.ca1Next = self.probeCA1(state)
        if reward == 1:
            self.learners[0].temporalDifference(reward)
            self.learners[0].updateCriticW()
            self.learners[0].lr = 0.01
            for i in range(100):
                self.learners[0].backward(state)
            for i in range(200):
                randomState = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
                self.ca1Next = self.probeCA1(randomState)
                self.learners[1].backward(self.ca1Next)
        elif reward == 0:
            self.learners[0].temporalDifference(reward)
            self.learners[0].lr = self.learners[0].TDdelta.item() * 0.1
            # self.learners[0].temporalDifference(reward)
            self.learners[0].updateCriticW()
            self.learners[0].backward(self.currState)
            self.navigation.learn(self.ca1Now, self.ca1Next, self.action)

        # Decay policy
        for i in range(delay):
            self.decayPolicy()

    def probeCA1(self, state):
        return self.learners[0].probeCA1(torch.Tensor(state)).detach().numpy()

    def decayPolicy(self):
        self.policyN = activatePolicy(self.policyN, self.tau)