import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

params = {
    "batch_size":128,
    "lr":1e-3,
    "discount_rate":0.8,    # important!!
    "train_epoch":500,
    "max_time":200,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class Net(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        hidden_list = [4, 8, 16, 8]
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_list[0]), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_list[0], hidden_list[1]), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_list[1], hidden_list[2]), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(hidden_list[2], hidden_list[3]), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(hidden_list[3], out_dim), nn.ReLU(True),nn.Softmax(out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Reinforce(object):
    def __init__(self, env):
        self.env = env
        self.policy = Net(2, 3)
        self.optimizer = optim.Adam(self.policy.parameters, lr=params["lr"])
        self.episode_rewards = []
        self.log_probabilities = []
        self.device = params["device"]

    def get_action_and_record(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probobility = self.policy(state)
        action_distribution = Categorical(action_probobility)
        action = action_distribution.sample().item()
        self.log_probabilities.append(action_distribution.log_prob(action))
        return action
    
    def train_model(self):
        for i_episode in range(params["train_epoch"]):
            state = self.env.reset()
            for t in range(params["max_time"]):
                action = self.get_action_and_record(state)
                state, reward, done, _ = self.env.step(action)
                self.episode_rewards.append(reward)
                if done:
                    self.update_policy()

    def update_policy(self):
        episode_G = np.zeros_like(self.episode_rewards)
        discounted_return = 0
        for i in range(len(self.episode_rewards)-1,-1,-1):
            discounted_return = discounted_return * params["discount_rate"] + self.episode_rewards[i]
            episode_G[i] = discounted_return
        
        # normalize episode rewards
        episode_G -= np.average(episode_G)
        episode_G /= np.std(episode_G)
