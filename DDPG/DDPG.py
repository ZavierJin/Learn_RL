import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

params = {
    "memory_size":10000,
    "batch_size":32,
    "actor_lr":1e-3,
    "critic_lr":1e-3,
    "tau":0.02,
    "discount_rate":0.99,
    "train_epoch":500,
    "max_time":200,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

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
        batch = self.transition(*zip(*random_transition))
        return batch

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        hidden_size = [256, 256]
        self.linear1 = nn.Linear(input_dim, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_dim)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        hidden_size = [256, 256]
        self.linear1 = nn.Linear(input_dim, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_dim)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPGAgent(object):

    def __init__(self, env):
        self.env = env
        self.num_observation = env.observation_space.shape[0]
        self.num_action = env.action_space.shape[0]
        self.device = params["device"]
        self.memory = ReplayMemory(params["memory_size"])
        self.batch_size = params["batch_size"]
        self.episode_durations = []

        self.actor = Actor(self.num_observation, self.num_action).to(self.device)
        self.actor_target = Actor(self.num_observation, self.num_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params["actor_lr"])

        self.critic = Critic(self.num_observation+self.num_action, self.num_action).to(self.device)
        self.critic_target = Critic(self.num_observation+self.num_action, self.num_action).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params["critic_lr"])

    def select_action(self, state):
        action = self.actor(state).detach()
        # action = torch.tensor(action.item(), device=self.device, dtype=torch.float)
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        next_state = torch.cat(batch.next_state)
        reward = torch.cat(batch.reward)

        # optimize critic net
        Q_value = self.critic(state, action)
        next_action = self.actor_target(next_state).detach()    # expected use target net
        expected_Q_value = reward + params["discount_rate"] * self.critic_target(next_state, next_action).detach() # detach from the graph
        criterion = nn.MSELoss()
        loss = criterion(Q_value, expected_Q_value)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # optimize actor net
        loss = - torch.mean(self.critic(state, self.actor(state)))
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # soft update target net
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - params["tau"]) + param.data * params["tau"])
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - params["tau"]) + param.data * params["tau"])

    def train_model(self):
        for i_episode in range(params["train_epoch"]):
            episode_reward = 0
            state = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float)
            for t in range(params["max_time"]):
                # Select and perform an action
                action = self.select_action(state)
                action_arr = np.array([action.item()])
                self.env.render(mode='rgb_array')

                # Observe new state
                next_state, reward, done, _ = self.env.step(action_arr)
                episode_reward += reward
                reward = torch.tensor(reward, device=self.device, dtype=torch.float)
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)

                # Store the transition in memory
                self.memory.push(state.view(1,-1), action.view(1,-1), next_state.view(1,-1), reward.view(1,1))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if done:
                    print("done.")
                    break        
            self.episode_durations.append(episode_reward)
            self.plot_durations()

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == "__main__":
    # env = gym.make('CartPole-v0').unwrapped     # unwrapped ?
    env = gym.make('Pendulum-v0').unwrapped
    plt.ion()   # Turn the interactive mode on
    agent = DDPGAgent(env)
    agent.train_model()
    
