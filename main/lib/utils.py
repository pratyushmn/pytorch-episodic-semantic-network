import numpy as np
import torch

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activatePolicy(policy, tau):
    return np.clip(policy * np.exp(-1 / tau) , 0, 1)

def convertToProbability(x, NewMin, NewMax):
    OldMin = torch.min(x)
    OldMax = torch.max(x)
    new_x = (((x - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    probs = new_x / torch.sum(new_x)
    return probs.cpu().numpy()

def convertToProbabilityNew(x, alpha, trialTime):
    new_alpha = max(alpha - trialTime/4000, 0)
    new_x = new_alpha*(x/np.sum(x)) + (1 - new_alpha)/8

    return new_x, new_alpha