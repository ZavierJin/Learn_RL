
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

params = {
    "memory_size":5000,
    "batch_size":128,
    "lr":1e-2,
    "max_epsilon":0.9,
    "min_epsilon":0.05,
    "epsilon_decay":200,
    "discount_rate":0.8,    # important!!
    "train_epoch":500,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        """ 
        For random sampling, the conversion of established batches is decorrelated. 
        Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        detailed explanation). This converts batch-array of Transitions to 
        Transition of batch-arrays.
        """
        random_transition = random.sample(self.memory, batch_size)
        return random_transition

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        hidden_list = [16, 32, 32]
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_list[0]), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_list[0], hidden_list[1]), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_list[1], hidden_list[2]), nn.ReLU(True))
        self.layer4 = nn.Linear(hidden_list[-1], out_dim)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DQNAgent(object):

    def __init__(self, env):
        self.env = env
        self.device = params["device"]
        self.memory = ReplayMemory(params["memory_size"])
        self.batch_size = params["batch_size"]
        self.discount = params["discount_rate"]
        self.eps = params["max_epsilon"]
        self.eps_step = 0
        self.resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC),T.ToTensor()])
        self.num_action = env.action_space.n
        self.num_observation = env.observation_space.shape[0]
        self.dqn_net = DQN(self.num_observation, self.num_action).to(self.device)
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=params["lr"])
        self.episode_durations = []
        

    def select_action(self, state):
        sample = random.random()
        self.eps = params["min_epsilon"] + (params["max_epsilon"] - params["min_epsilon"]) * \
            math.exp(-1. * self.eps_step / params["epsilon_decay"])
        self.eps_step += 1
        if sample > self.eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.dqn_net(state)
                return torch.argmax(action).view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_action)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to dqn_net
        state_action_values = self.dqn_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.dqn_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.view(-1, 1) * self.discount) + reward_batch
            
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_model(self):
        for i_episode in range(params["train_epoch"]):
            # Initialize the environment and state
            state = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                self.env.render(mode='rgb_array')

                # Observe new state
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
                if done:
                    reward = torch.tensor([-1], device=self.device, dtype=torch.float32)
    
                # Store the transition in memory
                self.memory.push(state.view(1,-1), action, next_state.view(1,-1), reward.view(1,1))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
        self.env.render()
        self.env.close()
        plt.ioff()
        plt.show()

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())


if __name__ == "__main__":
    env = gym.make('CartPole-v0').unwrapped     # unwrapped ?
    plt.ion()   # Turn the interactive mode on
    agent = DQNAgent(env)
    agent.train_model()
