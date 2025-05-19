import torch
from torch import nn
from torch.nn import functional as F


class DQN_model(nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super(DQN_model, self).__init__()
        self.lin1 = nn.Linear(state_dim, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class Agent:
    def __init__(self):
        self.model = DQN_model()
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()

    def act(self, state):
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_vals = self.model(state_v)
        action = q_vals.argmax(dim=1).item()
        return action