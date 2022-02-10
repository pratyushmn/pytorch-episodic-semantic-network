import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activatePolicy(policy, tau):
    return np.clip(policy * np.exp(-1 / tau) , 0, 1)

def convertToProbability(x, NewMin, NewMax):
    OldMin = np.amin(x)
    OldMax = np.amax(x)
    new_x = (((x - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return new_x / np.sum(new_x)

def convertToProbabilityNew(x, alpha, trialTime):
    new_alpha = max(alpha - trialTime/4000, 0)
    new_x = new_alpha*(x/np.sum(x)) + (1 - new_alpha)/8

    return new_x, new_alpha