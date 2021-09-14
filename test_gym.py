import time
import gym
from gym.spaces.box import Box
import torch

# x = torch.randn(2).view(1,-1)
# print(x)
# x = torch.cat((x,x,x), 0)
# print(x)

# env = gym.make('CartPole-v0')
env = gym.make('Pendulum-v0')
print(env.action_space)
print(env.observation_space.shape[0])
# for i_episode in range(2):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             # env.reset()
#             break
#     # time.sleep(0.1)
# print("finish")
# env.close()