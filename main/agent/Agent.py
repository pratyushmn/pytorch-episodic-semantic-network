import numpy as np
import torch
from .components.EpisodicLearner import EpisodicLearner
from .components.SemanticLearner import SemanticLearner
from .components.NavigationLearner import NavigationLearner
from ..lib.utils import activatePolicy

class Agent():
    def __init__(self, units: int = 980, contextDimension: int = 0, actionSpace: int = 8, episodic: bool = True, semantic: bool = True, priorKnowledge: bool = True) -> None:
        """Initializes an agent class that can learn to correctly choose actions given an input state, using a combination of 
        episodic and semantic memory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        self.episodicLearner = EpisodicLearner(self.device, units, numContext=contextDimension)
        self.semanticLearner = SemanticLearner(self.device, units)
        self.navigationLearner = NavigationLearner(self.device, units, actionSpace=actionSpace)

        # Episodic, semantic, and overall
        self.goals = [np.zeros(units), np.zeros(units), np.zeros(units)]

        # Policy to switch between episodic and semantic network
        self.beta = 1000
        self.policyN = 1
        self.episodic = episodic
        self.semantic = semantic
        self.priorKnowledge = priorKnowledge

        # last action
        self.action = 0

    def act(self, state: np.ndarray, trialTime: int) -> int:
        """Returns an action to choose based on input state and current number of steps taken.
        """
        state = torch.Tensor(state).to(self.device)
        self.ca1Now = self.probeCA1(state)
        self.currState = state

        self.goals[0] = self.episodicLearner(state)
        self.episodicLearner.predict_state_val(state)
        self.goals[1] = self.semanticLearner(self.ca1Now)

        # Overall goal
        if self.episodic and self.semantic:
            self.goals[2] = (self.policyN * self.goals[0] + (1 - self.policyN)  * self.goals[1])
        elif self.episodic:
            self.goals[2] = self.goals[0]
        elif self.semantic: 
            self.goals[2] = self.goals[1]

        # Choose an action based on goal and predictions from nav network
        self.action = self.navigationLearner.choose(self.goals[2], self.ca1Now, trialTime)

        # Decay policy
        self.decayPolicy()

        return self.action

    def learn(self, state: np.ndarray, reward: float) -> None:
        """Updates weights of component networks.
        """
        state = torch.Tensor(state).to(self.device)
        self.ca1Next = self.probeCA1(state)

        self.navigationLearner.learn(self.ca1Now, self.ca1Next, self.action)
        self.teachEpisodicLearner(state, reward, 50 if reward > 0 else 1)
        if reward > 0: self.teachSemanticLearner(state, 200)

    def teachEpisodicLearner(self, state: torch.Tensor, reward: int, epochs: int):
        """Updates weights of episodic learner.
        """
        self.episodicLearner.temporalDifference(reward)
        self.episodicLearner.updateCriticWeights()
        
        if reward > 0: self.episodicLearner.updateLearningRate(0.1)
        else: self.episodicLearner.updateLearningRate(0.1*self.episodicLearner.returnCurrentStateVal())

        for _ in range(epochs): self.episodicLearner.backward(state)

    def teachSemanticLearner(self, randomState: torch.Tensor, epochs: int) -> None:
        """Updates weights of semantic learner.
        """
        for _ in range(epochs):
            self.semanticLearner.backward(self.episodicLearner(randomState))
            if self.priorKnowledge is True: 
                x, y = np.random.uniform(-15, 15), np.random.uniform(-15, 15)
                c = 1 if x <= 0 else 2
                randomState = [x, y, c]
            else: randomState = [np.random.uniform(-15, 15), np.random.uniform(-15, 15), 0]
            randomState = torch.Tensor([(randomState[0] + 15)/30, (randomState[1] + 15)/30, randomState[2]/2]).to(self.device)

    def probeCA1(self, state: torch.Tensor) -> torch.Tensor:
        """Returns place cell representation of current state.
        """
        return self.episodicLearner.probeCA1(state)

    def decayPolicy(self) -> None:
        """Updates the balance between episodic and semantic memory goal every step.
        """
        self.policyN = activatePolicy(self.policyN, self.beta)