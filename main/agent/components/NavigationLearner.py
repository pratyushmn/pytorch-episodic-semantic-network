import torch
import torch.nn as nn
import numpy as np
from ...lib.utils import convertToProbability

class AddOne(nn.Module):
    def __init__(self) -> None:
        """Initializes a nn module which adds 1 to all elements of input tensor. Not sure why this is needed, but this operation was done in the original code everytime before the sigmoid function was applied. 
        """
        super(AddOne, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds 1 to all elements in input tensor.
        """
        return x + 1

class NavigationLearner(nn.Module):
    def __init__(self, units: int, lr: float = 0.05) -> None:
        """Initializes a navigation learner class that predicts the next state of the environment given a combination of episodic + semantic memory outputs of the current state of the environment, for all possible actions.
        """
        super(NavigationLearner, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.constant_(m.bias, 0)

        # In the original code, there's no bias in the linear layers. 
        # I decided to keep a bias but initialize it to zero. However, bias can be removed from th elinear layers by also adding a bias=False parameter to the nn.Linear constructor
        self.evaluator = nn.Sequential(
            nn.Linear(units + 8, 100),
            AddOne(),
            nn.Sigmoid(),
            nn.Linear(100, units),
            AddOne(),
            nn.Sigmoid()
        )

        self.evaluator.apply(init_weights)

        # self.W0 = nn.Parameter(torch.Tensor(np.random.normal(0, 0.1, (units + 8, 100))))
        # self.W1 = nn.Parameter(torch.Tensor(np.random.normal(0, 0.1, (100, units))))

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of the neural network for a given input.
        """
        # x = torch.sigmoid(torch.matmul(x, self.W0) + 1)
        # return torch.sigmoid(torch.matmul(x, self.W1) + 1)
        return self.evaluator(x)

    def choose(self, state: np.ndarray, memoryGoal: np.ndarray, trialTime: int) -> int:
        """Returns an action to choose based on CA1 state and current memory of goal.
        """
        hamming = np.zeros(8)
        actions = np.zeros(8)

        for i in range(8):
            actions.fill(0)
            actions[i] = 1
            
            x = torch.Tensor(np.append(state, actions))

            output = self.forward(x).detach().numpy()

            hamming[i] = np.sum(np.absolute(memoryGoal - output))

        # To prevent error in convertToProbability function when all values of the hamming array are equal, manually assign equal probabilities to all actions in that case.
        # For some reason, all equal hamming array only happens when you switch to using MSE Loss + Adam optimizer steps, instead of manually updating weights.
        # I tried using matrix multiplication with Pytorch instead of Numpy, and hamming array was never all equal until I switched to using an optimizer instead of manual weight updates.
        if np.all(hamming == hamming[0]): 
            print("All hamming are the same at trialtime {}".format(trialTime))
            prob = np.ones(8)/8

        else: prob = convertToProbability(1 - hamming, np.clip(trialTime/2000, 0, 0.99), 1)

        return np.random.choice(np.arange(8), p=prob)

    def learn(self, state: np.ndarray, next_state: np.ndarray, action: int) -> None:
        """Updates neural network weights.
        """
        self.optimizer.zero_grad()

        actions = np.zeros(8)
        actions[action] = 1

        x = torch.Tensor(np.append(state, actions))
        next_state = torch.Tensor(next_state)

        state_prediction = self.forward(x)

        loss = self.loss(state_prediction, next_state)

        loss.backward()
        self.optimizer.step()