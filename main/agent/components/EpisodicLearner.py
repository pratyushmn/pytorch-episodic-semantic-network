import numpy as np
import torch
import torch.nn as nn
from itertools import chain

class AutoEncoder(nn.Module):
    def __init__(self, units: int) -> None:
        """Initializes a nn module "autoencoder" which puts the input through the same layer twice. 
        """
        super(AutoEncoder, self).__init__()

        self.CA3W = nn.Linear(units, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of the "autoencoder" for a given input. 1 is subtracted elementwise everytime before the sigmoid function is applied in the original code; not sure why.
        """
        step1 = torch.sigmoid(self.CA3W(x))
        step2 = torch.sigmoid(self.CA3W(step1))
        return step2

class Critic(nn.Module):
    def __init__(self, units: int, gamma: float) -> None:
        """Initializes a nn module "critic" which acts as a value function for the input state. 
        """
        super(Critic, self).__init__()

        self.critic_layer = nn.Linear(units, 1)
        self.gamma = gamma

        self.last_val = None
        self.curr_val = None
        self.TDdelta = 0

        self.optim = torch.optim.SGD(self.critic_layer.parameters(), lr=0.04)

    def activateCritic(self, spatial_CA1: torch.Tensor) -> None:
        """Saves the critic's valuation of the input spatially cued CA1.
        """
        self.last_val = self.curr_val
        self.curr_val = torch.tanh(self.critic_layer(spatial_CA1)) 

    def temporalDifference(self, reward: int) -> None:
        """Computes critic prediction error based on the reward and past predictions, using temporal difference learning strategies.
        """
        if self.last_val is not None:
            self.TDdelta = reward + self.gamma * self.curr_val - self.last_val

    def updateCriticW(self) -> None:
        """Updates weights for critic based on TD delta.
        """
        # can only do TD learning if at least 2 states have been experienced
        if self.last_val is None: return

        # update the learning rate based on the TD delta
        for g in self.optim.param_groups: g['lr'] = 0.04*self.TDdelta.item()

        # backprop
        self.optim.zero_grad()
        critic_loss = nn.functional.l1_loss(self.last_val, self.TDdelta + self.last_val)
        critic_loss.backward(retain_graph=True)
        self.optim.step()

class EpisodicLearner(nn.Module):
    def __init__(self, device, units: int, lr: float = 0.01, gamma: float = 0.95, place_field_breadth: float = 0.16, numContext: int = 0) -> None:
        """Initializes an episodic learner class.
        """
        super(EpisodicLearner, self).__init__()

        self.CA3Units = units
        self.CA1Units = units
        self.device = device

        self.CA1Fields = [torch.empty(self.CA1Units, device=self.device).uniform_(0, 1) for i in range(numContext + 2)]

        # NN Layers
        self.input_to_autoencoder = nn.Linear(numContext + 2, self.CA3Units)
        self.autoencoder = AutoEncoder(self.CA3Units)
        self.autoencoder_to_linear = nn.Linear(self.CA3Units, self.CA1Units)
        self.critic = Critic(self.CA1Units, gamma)
        
        self.input_to_autoencoder.to(self.device)
        self.autoencoder.to(device)
        self.autoencoder_to_linear.to(self.device)
        self.critic.to(self.device)

        self.lr = lr
        
        self.place_field_breadth = place_field_breadth

        self.loss = nn.L1Loss()

        self.CA3_optimizer = torch.optim.SGD(chain(self.input_to_autoencoder.parameters(), self.autoencoder.parameters()), lr=self.lr, momentum=0.5)
        self.CA1_optimizer = torch.optim.SGD(self.autoencoder_to_linear.parameters(), lr=self.lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return CA1 output cued from CA3 for given state; also update critic-based state values using CA1 cued from spatial input
        """
        # CA3   
        x = torch.sigmoid(self.input_to_autoencoder(state))
        CA3 = self.autoencoder(x)

        # CA1 cued from the CA3
        CA1_from_CA3 = torch.sigmoid(self.autoencoder_to_linear(CA3))
        
        return CA1_from_CA3.detach()

    def predict_state_val(self, state: torch.Tensor) -> None:
        """Compute the critic prediction of state value and save it.
        """
        # CA1 cued from space
        CA1_from_spatial = self.probeCA1(state)

        # Critic value for state based on CA1 cued from space
        self.critic.activateCritic(CA1_from_spatial)

    def backward(self, state: torch.Tensor) -> None:
        """Train the weights of all neural network layers (except the critic) based on the input state.
        """

        # updating learning rates (ie. self.lr is 0.1 if reward was found, else 0.1*self.curr_val)
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
        """Outputs the spatially cued CA1 (place cells) based on input state.
        """
        return (torch.exp(-(sum([(state[i] - self.CA1Fields[i])**2 for i in range(len(state))])) / (2 * (self.place_field_breadth**2))))