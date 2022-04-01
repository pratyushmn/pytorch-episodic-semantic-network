import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from ...lib.utils import convertToProbability, convertToProbabilityNew

class NavigationLearner(nn.Module):
    def __init__(self, units: int, lr: float = 0.01, actionSpace: int = 8) -> None:
        """Initializes a navigation learner class that predicts the next state of the environment given a combination of episodic + semantic memory outputs of the current state of the environment, for all possible actions.
        """
        super(NavigationLearner, self).__init__()

        # In the original code, there's no bias in the linear layers. 
        # I decided to keep a bias but initialize it to zero. However, bias can be removed from th elinear layers by also adding a bias=False parameter to the nn.Linear constructor
        self.evaluator = nn.Sequential(
            nn.Linear(units + actionSpace, 100),
            nn.Sigmoid(),
            nn.Linear(100, units),
            nn.Sigmoid()
        )

        self.lr = lr
        self.eps = 1
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()

        self.actionSpace = actionSpace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of the neural network for a given input.
        """
        # x = torch.sigmoid(torch.matmul(x, self.W0) + 1)
        # return torch.sigmoid(torch.matmul(x, self.W1) + 1)
        return self.evaluator(x)

    def choose(self, memoryGoal: np.ndarray, state: np.ndarray, trialTime: int) -> int:
        """Returns an action to choose based on CA1 state and current memory of goal.
        """
        hamming = np.zeros(self.actionSpace)
        norms = np.zeros(self.actionSpace)
        actions = np.zeros(self.actionSpace)

        for i in range(self.actionSpace):
            actions.fill(0)
            actions[i] = 1

            x = torch.Tensor(np.append(state, actions))

            output = self.forward(x).detach().numpy()
            hamming[i] = np.sum(np.absolute(memoryGoal - output))
            norms[i] = 1/np.sum(np.absolute(memoryGoal - output))

        if np.all(hamming == hamming[0]): 
            print("All Hammings are equal.")
            probs = np.array([1/self.actionSpace for i in range(self.actionSpace)])
        else: probs = convertToProbability(1 - hamming, np.clip(trialTime/2000, 0, 0.99), 1)

        # else: probs = convertToProbability(1 - hamming, np.clip(trialTime/2000, 0, 0.99), 1)
        # else: probs, self.eps = convertToProbabilityNew(norms, self.eps, trialTime)

        # return Categorical(torch.Tensor(probs)).sample()
        return np.random.choice(np.arange(self.actionSpace), p=probs)

    def learn(self, state: np.ndarray, next_state: np.ndarray, action: int) -> None:
        """Updates neural network weights.
        """
        self.optimizer.zero_grad()

        actions = np.zeros(self.actionSpace)
        actions[action] = 1

        x = torch.Tensor(np.append(state, actions))
        next_state = torch.Tensor(next_state)

        state_prediction = self.forward(x)

        loss = self.loss(state_prediction, next_state)
        # print("Loss: {}".format(loss.item()*980))

        loss.backward()
        self.optimizer.step()