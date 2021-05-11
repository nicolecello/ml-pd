import gym
import random
import numpy as np
import matplotlib as plt
import sys
sys.path.insert(1, './bipedal-walker/ddpg-torch/')
import train


env = gym.make("BipedalWalkerHardcore-v3")
observation = env.reset()
action_size = env.action_space.shape[0]
# Random Agent
def random_agent(episodes):
  for _ in range(episodes):
    env.render()
    action = np.random.uniform(-1.0,1.0,size=action_size)
    observation, reward, done, info = env.step(action)
    if done: 
      observation = env.reset()

# Engrained Policy (Might also be random)
def default_agent(episodes):
  for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    print(f'Action: {action}')
    observation, reward, done, info = env.step(action)
    print(f'observation: {observation}')

  if done:
    print("done")
    observation = env.reset()

def ddpg_agent():
  scores = train.ddpg(episodes=100, step=2000, pretrained=1, noise=0)
  fig = plt.figure()
  plt.plot(np.arange(1, len(scores) + 1), scores)
  plt.ylabel('Score')
  plt.xlabel('Episode #')
  plt.show()

env.close()
# MODELS 
# random_agent(1000)
# default_agent(1000)
ddpg_agent()
