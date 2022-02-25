import numpy as np
import torch
import torch.nn as nn

class SubtractOne(nn.Module):
    def __init__(self) -> None:
        """Initializes a nn module which subtracts 1 from all elements of input tensor. Not sure why this is needed, but this operation was done in the original code before the sigmoid function was applied for the autoencoder. 
        """
        super(SubtractOne, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtracts 1 elementwise from all elements in input tensor.
        """
        return x - 1

class AutoEncoder(nn.Module):
    def __init__(self, units: int) -> None:
        """Initializes a nn module "autoencoder" which puts the input through the same layer twice. 
        """
        super(AutoEncoder, self).__init__()

        self.CA3W = nn.Linear(units, units)
        self.subtractOne = SubtractOne()

        # initialize weights in the same way done in the original code (although original code didn't have any bias at all)
        torch.nn.init.normal_(self.CA3W.weight, mean=0.0, std=0.1)
        torch.nn.init.constant_(self.CA3W.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of the "autoencoder" for a given input. 1 is subtracted elementwise everytime before the sigmoid function is applied in the original code; not sure why.
        """
        step1 = torch.sigmoid(self.subtractOne(self.CA3W(x)))
        step2 = torch.sigmoid(self.subtractOne(self.CA3W(step1)))
        return step2

class EpisodicLearner(nn.Module):
    def __init__(self, units: int, lr: float = 0.01, gamma: float = 0.95, place_field_breadth: float = 0.16, numContext: int = 0) -> None:
        """Initializes an episodic learner class.
        """
        super(EpisodicLearner, self).__init__()

        self.CA3Units = units
        self.CA1Units = units

        self.CA1Fields = [torch.empty(self.CA1Units).uniform_(0, 1) for i in range(numContext + 2)]

        # Stored activity patterns
        self.state_vals = [0, 0] # critic values for last state and current state
        self.TDdelta = 0

        # NN Layers
        self.input_to_autoencoder = nn.Linear(numContext + 2, self.CA3Units)
        self.autoencoder = AutoEncoder(self.CA3Units)
        self.autoencoder_to_linear = nn.Linear(self.CA3Units, self.CA1Units)
        self.critic_layer = nn.Linear(self.CA1Units, 1)

        # initialize weights in the same way done in the original code (although original code didn't have any bias at all)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.constant_(m.bias, 0)

        self.input_to_autoencoder.apply(init_weights)
        self.autoencoder_to_linear.apply(init_weights)
        self.critic_layer.apply(init_weights)

        self.lr = lr
        self.gamma = gamma
        self.place_field_breadth = place_field_breadth

        self.loss = nn.MSELoss()

        self.CA3_optimizer = torch.optim.SGD(self.autoencoder.parameters(), lr=self.lr, momentum=0.5)
        self.CA1_optimizer = torch.optim.SGD(self.autoencoder_to_linear.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.SGD(self.critic_layer.parameters(), lr=self.lr)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Return CA1 output cued from CA3 for given state; also update critic-based state values using CA1 cued from spatial input
        """
        state = torch.Tensor(state)

        # CA3   
        x = torch.sigmoid(self.input_to_autoencoder(state))
        CA3 = self.autoencoder(x)

        # CA1 cued from the CA3
        CA1_from_CA3 = torch.sigmoid(self.autoencoder_to_linear(CA3))
        
        return CA1_from_CA3.detach().numpy()

    def predict_state_val(self, state: np.ndarray) -> None:
        """Compute the critic prediction of state value and save it.
        """
        # CA1 cued from space
        CA1_from_spatial = self.probeCA1(state)

        # Critic value for state based on CA1 cued from space
        self.state_vals[0] = self.state_vals[1]
        self.state_vals[1] = self.activateCritic(CA1_from_spatial)

    def backward(self, state: np.ndarray) -> None:
        """Train the weights of all neural network layers (except the critic) based on the input state.
        """
        state = torch.Tensor(state)

        # updating learning rates (ie. self.lr is 0.01 if reward was found, else 0.1*self.TDdelta)
        for g in self.CA3_optimizer.param_groups: g['lr'] = self.lr 
        for g in self.CA1_optimizer.param_groups: g['lr'] = self.lr 

        x = torch.sigmoid(self.input_to_autoencoder(state))

        # CA3 learning
        self.CA3_optimizer.zero_grad()
        CA3 = self.autoencoder(x)
        CA3_loss = self.loss(CA3, x)
        CA3_loss.backward()
        self.CA3_optimizer.step()
        
        # CA1 learning
        self.CA1_optimizer.zero_grad()
        CA1_from_CA3 = torch.sigmoid(self.autoencoder_to_linear(CA3.detach()))
        CA1_from_spatial = self.probeCA1(state)
        CA1_loss = self.loss(CA1_from_CA3, CA1_from_spatial)
        CA1_loss.backward()
        self.CA1_optimizer.step()

    def probeCA1(self, state: torch.Tensor) -> torch.Tensor:
        """Outputs the spatially cued CA1 based on input state.
        """
        return (torch.exp(-((state[0] - self.CA1Fields[0])**2  + (state[1] - self.CA1Fields[1])**2) / (2 * (self.place_field_breadth**2))))

    def activateCritic(self, spatial_CA1: torch.Tensor) -> torch.Tensor:
        """Outputs the critic's valuation of the input spatially cued CA1.
        """
        # sigmoid wasn't originally there in paper 
        # but I added it since it makes sense to have it here (prevent exploding predictions)
        return torch.sigmoid(self.critic_layer(spatial_CA1)) 

    def temporalDifference(self, reward: int) -> None:
        """Computes critic prediction error based on the reward and past predictions, using temporal difference learning strategies.
        """
        #  since reward is only non-zero when trial ends, as such V(s_{t + 1}) will be 0
        # if the trial hasn't ended, then V(s_{t + 1}) is non-zero but then reward must be 0
        if reward == 1:
            self.TDdelta = reward - self.state_vals[0]
        else:
            self.TDdelta = self.gamma * self.state_vals[1] - self.state_vals[0]

    def updateCriticW(self) -> None:
        """Updates weights for critic based on TD delta.
        """
        # can only do TD learning if at least 2 states have been experienced
        if isinstance(self.state_vals[0], int): return 

        # update the learning rate based on the TD delta
        for g in self.critic_optimizer.param_groups: g['lr'] = 0.04*self.TDdelta.item()

        # backprop
        self.critic_optimizer.zero_grad()
        critic_loss = -1* self.state_vals[0]
        critic_loss.backward()
        self.critic_optimizer.step()