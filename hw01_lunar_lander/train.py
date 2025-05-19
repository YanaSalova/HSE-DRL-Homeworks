from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy
from collections import deque, namedtuple


GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
SEED = 42

class DQN_model (nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super(DQN_model, self).__init__()
        self.lin1 = nn.Linear(state_dim, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class ReplayBuffer():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, idx):
        samples = [self.memory[i] for i in idx]
        return zip(*samples)



class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        self.model = DQN_model(state_dim, action_dim) # Torch model
        self.buffer = ReplayBuffer(capacity=100000)
        self.Transition = namedtuple('Transition',
                                     ('state',
                                      'action',
                                      'next_state',
                                      'reward',
                                      'done'))
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.push(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        idx = np.random.randint(0, len(self.buffer), BATCH_SIZE)
        state, action, next_state, reward, done = self.buffer.sample(idx)
        state      = torch.tensor(np.array(state)).float()
        next_state = torch.tensor(np.array(next_state)).float()
        reward     = torch.tensor(np.array(reward)).float()
        action     = torch.tensor(np.array(action), dtype=torch.int64)
        done       = torch.tensor(np.array(done))

        return (state, next_state, reward, action, done)


    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, next_state, reward, action, done = batch

        with torch.no_grad():
            next_q = self.target_model(next_state).max(1)[0]
            target = reward + (1 - done.float()) * GAMMA * next_q

        current_q = self.model(state).gather(1, action.view(-1,1)).squeeze(1)
        loss = self.loss(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = np.array(state)
        return self.model(torch.tensor(state).float()).max(dim=0)[1].item()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state, _  = env.reset()
        total_reward = 0.
        
        while not done:
             next_state, reward, terminated, truncated, _ = env.step(agent.act(state))
             done = terminated or truncated
             state = next_state
             total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state, _ = env.reset()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        #next_state, reward, done, _ = env.step(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()[0]
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        #next_state, reward, done, _ = env.step(action)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()[0]
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            dqn.save()
