import gym
import random
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import sys
import importlib
# ppoModule =  input("ppo PPOAgent")
# importlib.import_module(ppoModule)
sys.path.insert(1, './ppo-model/ppo.py')
sys.path.insert(1, './ddpg-model/ddpg-torch/')
import train
from ddpg_agent import Agent
from ppomodel import ppo
# import ppo


env = gym.make("BipedalWalker-v3")
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
    next_state, reward, done, info = env.step(action)
    print(f'observation: {next_state}')

  if done:
    print("done")
    observation = env.reset()

def ddpg_agent():
  state_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0)
  # agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0)
  for _ in range(1000):
    # initial state
    state = env.reset()
    env.render()
    action = agent.act(state) # ddpg actions
    print(f'Action: {action}')
    next_state, reward, done, info = env.step(action[0])
    print(info)
    print(done)
    while not done:
      action = agent.act(next_state)
      next_state, reward, done, info = env.step(action[0])
      print(f'observation: {observation}')
    env.reset()
  # Actor.act()
  scores = train.ddpg(episodes=100, step=2000, pretrained=1, noise=0)
  fig = plt.figure()
  fig.plot(np.arange(1, len(scores) + 1), scores)
  fig.ylabel('Score')
  fig.xlabel('Episode #')
  fig.show()

  

env.close()
# MODELS 
# random_agent(1000)
# default_agent(1000)
# ddpg_agent()
env_name = 'BipedalWalkerHardcore-v3'
ppo = ppo.PPOAgent(env_name,'Teachers/','Agent_2')
ppo.test()
# ppo.act(env)
# ddpg_agent()