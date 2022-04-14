import numpy as np
import torch
from .components import EpisodicLearner
from .components import SemanticLearner
from .components import NavigationLearner
from ..lib.utils import activatePolicy

class StandardAgent():
    def __init__(self, episodicUnits: int = 980, contextDimension: int = 0, actionSpace: int = 8, episodic: bool = True, semantic: bool = True, priorKnowledge: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))
        
        self.learners = []
        self.learners.append(EpisodicLearner.EpisodicLearner(self.device, episodicUnits, numContext=contextDimension))
        self.learners.append(SemanticLearner.SemanticLearner(self.device, episodicUnits))
        self.navigation = NavigationLearner.NavigationLearner(self.device, episodicUnits, actionSpace=actionSpace)

        self.priorKnowledge = priorKnowledge

        # Episodic, semantic, and overall
        self.goals = [np.zeros(episodicUnits), np.zeros(episodicUnits), 
                np.zeros(episodicUnits)]

        # Policy to switch between episodic and semantic network
        self.tau = 2000
        self.policyN = 1
        self.episodic = episodic
        self.semantic = semantic

        # last action
        self.action = 0

    def act(self, state: np.ndarray, trialTime: int):
        # Episodic goal
        state = torch.Tensor(state).to(self.device)
        self.ca1Now = self.probeCA1(state)
        self.currState = state

        self.goals[0] = self.learners[0].forward(state)
        self.learners[0].predict_state_val(state)
        
        self.goals[1] = self.learners[1].forward(self.ca1Now)

        # Overall goal
        if self.episodic and self.semantic:
            self.goals[2] = (self.policyN * self.goals[0] + (1 - self.policyN)  * self.goals[1])
        elif self.episodic:
            self.goals[2] = self.goals[0]
        elif self.semantic: 
            self.goals[2] = self.goals[1]

        # Choose an action based on goal and predictions from nav network
        self.action = self.navigation.choose(self.goals[2], self.ca1Now, trialTime)

        # Decay policy
        self.decayPolicy()

        return self.action

    def learn(self, state: np.ndarray, reward: float):
        # We need to take a step in the environment
        # before we learn so that the navigation net has
        # access to the next state given the action taken
        state = torch.Tensor(state).to(self.device)
        self.ca1Next = self.probeCA1(state)
        if reward > 0:
            self.learners[0].temporalDifference(reward)
            self.learners[0].updateCriticW()
            self.learners[0].lr = 0.1
            for i in range(50):
                self.learners[0].backward(state)

            randomState = state
            for i in range(200):
                self.ca1Next = self.learners[0].forward(randomState)
                self.learners[1].backward(self.ca1Next)
                if self.priorKnowledge is True: 
                    x = np.random.uniform(-15, 15)
                    y = np.random.uniform(-15, 15)
                    c = 1 if x <= 0 else 2
                    randomState = [x, y, c]
                else: randomState = [np.random.uniform(-15, 15), np.random.uniform(-15, 15), 0]

                randomState = torch.Tensor([(randomState[0] + 15)/30, (randomState[1] + 15)/30, randomState[2]/2]).to(self.device)
        else:
            self.learners[0].temporalDifference(reward)
            self.learners[0].lr = self.learners[0].curr_val.item() * 0.1
            self.learners[0].updateCriticW()
            self.learners[0].backward(self.currState)

            self.navigation.learn(self.ca1Now, self.ca1Next, self.action)

    def probeCA1(self, state: torch.Tensor):
        return self.learners[0].probeCA1(state)

    def decayPolicy(self):
        self.policyN = activatePolicy(self.policyN, self.tau)