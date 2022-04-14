import numpy as np
import torch

def activatePolicy(policy, tau):
    return np.clip(policy * np.exp(-1 / tau) , 0, 1)

def convertToProbability(x, NewMin, NewMax):
    OldMin = torch.min(x)
    OldMax = torch.max(x)
    new_x = (((x - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    probs = new_x / torch.sum(new_x)
    return probs.cpu().numpy()