import random
import numpy as np
import os
import torch
from torch.distributions import Normal



class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl", map_location="cpu")
        
    def act(self, state):
        with torch.no_grad():
            mu = self.model(torch.tensor(np.array(state)).float())
        return torch.tanh(mu).cpu().numpy()   

    def reset(self):
        pass

