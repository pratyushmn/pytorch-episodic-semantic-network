import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class SemanticLearner(nn.Module):
    def __init__(self, device, vUnits: int, hUnits: int = 300, lr: float = 0.00001, k: int = 1) -> None:
        """Initializes a semantic learner class (based on Restricted Boltzmann Machines).
        """
        super(SemanticLearner, self).__init__()

        self.device = device 

        self.num_visible = vUnits
        self.num_hidden = hUnits
        self.lr = lr
        self.k = k # for contrastive divergence

        self.v = nn.Parameter(torch.Tensor(np.random.normal(0, 0.01, (1, self.num_visible))))
        self.h = nn.Parameter(torch.Tensor(np.random.normal(0, 0.01, (1, self.num_hidden))))
        self.W = nn.Parameter(torch.Tensor(np.random.normal(0, 0.01, (self.num_hidden, self.num_visible))))
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        self.to(self.device)

    def visible_to_hidden(self, v: torch.Tensor) -> torch.Tensor:
        """Conditional sampling of a hidden variable given a visible variable.
        """
        return torch.sigmoid(F.linear(v, self.W, self.h)).bernoulli()

    def hidden_to_visible(self, h: torch.Tensor) -> torch.Tensor:
        """Conditional distribution of a visible variable given a hidden variable.
        """
        return torch.sigmoid(F.linear(h, self.W.t(), self.v))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the generated state from the input state.
        """
        state = state.view(-1, self.num_visible)
        h = self.visible_to_hidden(state)

        # Contrastive Divergence
        for _ in range(self.k):
            v = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v.bernoulli())

        return v.detach().view(-1)

    def gibbs_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output of RBM (ie. visible units after passing through RBM k times).
        """
        h = self.visible_to_hidden(x)

        # Contrastive Divergence
        for _ in range(self.k):
            v = self.hidden_to_visible(h)
            h = self.visible_to_hidden(v.bernoulli())

        return v

    def free_energy(self, v: torch.Tensor) -> torch.FloatTensor:
        """Free energy function from Hwang et al. 2020
        """
        v_term = torch.matmul(v, self.v.t()) # matrix multiplication of input v and bias for visible layer
        sum_term = torch.sum(F.softplus(F.linear(v, self.W, self.h)), dim=1)
        return torch.mean(-v_term - sum_term) 
    
    def backward(self, x: torch.Tensor) -> None:
        """Train the weights of the RBM based on the input example state x.
        """
        self.optimizer.zero_grad()

        x = x.view(-1, self.num_visible)

        loss = self.free_energy(self.gibbs_sampling(x)) - self.free_energy(x)

        loss.backward()
        self.optimizer.step()