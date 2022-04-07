import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from ...lib.utils import convertToProbability, convertToProbabilityNew

class NavigationLearner(nn.Module):
    def __init__(self, device, units: int, lr: float = 0.0075, actionSpace: int = 8) -> None:
        """Initializes a navigation learner class that predicts the next state of the environment given a combination of episodic + semantic memory outputs of the current state of the environment, for all possible actions.
        """
        super(NavigationLearner, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.1)

        self.device = device

        # In the original code, there's no bias in the linear layers. 
        self.evaluator = nn.Sequential(
            nn.Linear(units + actionSpace, 100),
            nn.Sigmoid(),
            nn.Linear(100, units),
            nn.Sigmoid()
        )

        self.evaluator.to(self.device)

        # self.evaluator.apply(init_weights)

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
        hamming = torch.zeros(self.actionSpace, device=self.device)
        norms = torch.zeros(self.actionSpace, device=self.device)
        actions = torch.zeros(self.actionSpace, device=self.device)

        for i in range(self.actionSpace):
            actions.fill_(0)
            actions[i] = 1

            x = torch.cat((state, actions))
            x.to(self.device)

            output = self.forward(x).detach()
            hamming[i] = torch.sum(torch.abs(memoryGoal - output))
            norms[i] = 1/torch.sum(torch.abs(memoryGoal - output))

        if torch.all(hamming == hamming[0]): 
            print("All Hammings are equal.")
            probs = np.array([1/self.actionSpace for i in range(self.actionSpace)])
        else: probs = convertToProbability(1 - hamming, np.clip(trialTime/2000, 0, 0.99), 1)

        # else: probs = convertToProbability(1 - hamming, np.clip(trialTime/2000, 0, 0.99), 1)
        # else: probs, self.eps = convertToProbabilityNew(norms, self.eps, trialTime)

        # return Categorical(torch.Tensor(probs)).sample()
        return np.random.choice(np.arange(self.actionSpace), p=probs)

    def learn(self, state: torch.Tensor, next_state: torch.Tensor, action: int) -> None:
        """Updates neural network weights.
        """
        self.optimizer.zero_grad()

        actions = torch.zeros(self.actionSpace, device=self.device)
        actions[action] = 1

        x = torch.cat((state, actions))

        state_prediction = self.forward(x)

        loss = self.loss(state_prediction, next_state)
        # print("Loss: {}".format(loss.item()*980))

        loss.backward()
        self.optimizer.step()