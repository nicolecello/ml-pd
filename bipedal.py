import gym
import random
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import sys
import pickle
from collections import deque
import torch
import importlib
# ppoModule =  input("ppo PPOAgent")
# importlib.import_module(ppoModule)
sys.path.insert(1, './ppo-model/ppo.py')
# sys.path.insert(1, './ddpg-model/ddpg-torch/')
sys.path.insert(1, './dqnmodel/')
# import train
# from ddpg_agent import Agent
from ppomodel import ppo
from dqnmodel import dqn
# import ppo

env = gym.make("BipedalWalker-v3")
env_name = 'BipedalWalkerHardcore-v3'
BATCH_SIZE = 64
MAX_EPISODES = 10
MAX_REWARD = 300
MAX_STEPS = 2000 #env._max_episode_steps
BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
MEAN_EVERY = 2

EPSILON_DECAY = 0.001
EPSILON_MIN = 0.001




'''Store all teachers here:
  Agent 0 - Random. Does not care about input.
  Agent 1 (PPO Trained) - Slow starter but rapidly alternates betweenn front and back leg. Well balanced
  Agent 2 (PPO Trained) - Propels forward mostly using its back leg. Balances itself with the front leg. Looks like it will fall but doesn't. Slow
  Agent 3 (PPO Trained) - Nearly perfect. Runs very fast, but has slower leg alternation than agent 1. 
'''
teachers = ['random_agent','ppo_agent1','ppo_agent2','ppo_agent3']

# observation = env.reset()
# action_size = env.action_space.shape[0]

# AGENTS

# Agent 0 - Random agent. No input required
# Returns a random action
def random_agent():
  return np.random.uniform(-1.0,1.0,size=action_size)

# PPO Agents - Agents trained using PPO. 
# Input: the agent's number and state
# Output:
def ppo_agent_general(agent_no,state):
  modelname = f'Agent_{agent_no}'
  agent = ppo.PPOAgent(env_name,'Teachers/',modelname)
  agent.load()
  # predicted best action to take
  action = agent.Actor.predict(state)[0]
  return action

# def ddpg_agent():
#   state_dim = env.observation_space.shape[0]
#   action_dim = env.action_space.shape[0]
  # # Actor.act()
  # scores = train.ddpg(episodes=100, step=2000, pretrained=1, noise=0)
  # fig = plt.figure()
  # fig.plot(np.arange(1, len(scores) + 1), scores)
  # fig.ylabel('Score')
  # fig.xlabel('Episode #')
  # fig.show()

def ppo_agent(agent_no):
  modelname = f'Agent_{agent_no}'
  test_episodes = 100
  agent = ppo.PPOAgent(env_name,'Teachers/',modelname)
  agent.load()
  for e in range(101):
    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size[0]])
    done = False
    score = 0
    while not done:
        agent.env.render()
        action = agent.Actor.predict(state)[0]
        print(action)
        state, reward, done, _ = agent.env.step(action)
        state = np.reshape(state, [1, agent.state_size[0]])
        score += reward
        if done:
            average, SAVING = agent.PlotModel(score, e, save=False)
            print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
            break
    agent.env.close()
    

'''Return a list containing the actions that each teacher would next take
  given the current state.
  The actions of each teacher is an array of 4 values from -1..1 '''
def get_teacher_actions(state):
  actions = []
  for t in teachers:
    actions.append(t[0]())

# A lot of the code for the student comes from: 
# https://github.com/claudeHifly/BipedalWalker-v3/blob/master/DQN/improved_DQN_solution.py 
def student():
  start_episode = 0
  n_state_params = env.observation_space.shape[0]
  n_actions = env.action_space.shape[0]
  eps = 0.99

  agent = dqn.Agent(n_state_params, n_actions, 0, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
  DIR = 'Students'
  LOAD = False
  if LOAD:
      agent.epsilon = 0.001
      start_episode = 67300
      agent.qnetwork_local.load_state_dict(torch.load(DIR + 'checkpoint_local_ep' + str(start_episode) + '.pth', map_location="cpu"))
      agent.qnetwork_target.load_state_dict(torch.load(DIR + 'checkpoint_target_ep' + str(start_episode) + '.pth', map_location="cpu"))

  scores = []
  mean_scores = []
  last_scores = deque(maxlen=MEAN_EVERY)
  distances = []
  mean_distances = []
  last_distance = deque(maxlen=MEAN_EVERY)
  losses_mean_episode = []

  for ep in range(start_episode + 1, MAX_EPISODES + 1):
      state = env.reset()
      total_reward = 0
      total_distance = 0
      losses = []
      for t in range(MAX_STEPS):
          action = agent.act(state,eps)
          next_state, reward, done, _ = env.step(action)
          env.render()
          loss = agent.step(state, action, reward, next_state, done)
          if loss is not None:
              losses.append(loss)
          state = next_state
          total_reward += reward
          if reward != -100:
              total_distance += reward
          if done:
              break
      eps = max(EPSILON_MIN, EPSILON_DECAY * eps)
      agent.epsilon = eps

      if len(losses) >= 1:
          mean_loss = np.mean(losses)
          losses_mean_episode.append((ep, mean_loss))
      else:
          mean_loss = None

      print('\rEpisode: {}/{},\tScore: {:.2f},\tDistance: {:.2f},\tloss: {},\te:{:.2f}'.format(ep, MAX_EPISODES,
                                                                                          total_reward,
                                                                                          total_distance, mean_loss,
                                                                                          agent.epsilon), end="")
      scores.append(total_reward)
      distances.append(total_distance)
      last_scores.append(total_reward)
      last_distance.append(total_distance)
      mean_score = np.mean(last_scores)
      mean_distance = np.mean(last_distance)

      # record rewards dynamically
      FILE = 'record.dat'
      data = [ep, total_reward, total_distance, mean_loss, agent.epsilon]
      with open(FILE, "ab") as f:
          pickle.dump(data, f)

      if (mean_score >= 300):
          print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, mean_score))
          torch.save(agent.qnetwork_local.state_dict(), DIR + '/best/checkpoint_local_ep' + str(ep) + '.pth')
          torch.save(agent.qnetwork_target.state_dict(), DIR + '/best/checkpoint_target_ep' + str(ep) + '.pth')
          break

      # save model every MEAN_EVERY episodes
      if ((ep % MEAN_EVERY) == 0):
          print('\rEpisode: {}/{},\tMean Score: {:.2f},\tMean Distance: {:.2f},\tloss: {},\te:{:.2f}'.format(ep, MAX_EPISODES,
                                                                                    mean_score,
                                                                                    mean_distance, mean_loss,
                                                                                    agent.epsilon))
          torch.save(agent.qnetwork_local.state_dict(), DIR + '/checkpoint_local_ep' + str(ep) + '.pth')
          torch.save(agent.qnetwork_target.state_dict(), DIR + '/checkpoint_target_ep' + str(ep) + '.pth')
          mean_scores.append(mean_score)
          mean_distances.append(mean_distance)
          FILE = 'record_mean.dat'
          data = [ep, mean_score, mean_distance, mean_loss, agent.epsilon]
          with open(FILE, "ab") as f:
              pickle.dump(data, f)
  env.close()

# ACTOR 1: Lag in the first step - front balancer 

env.close()
student()
# get_teacher_actions()
# MODELS 
# random_agent(1000)
# default_agent(1000)
# ddpg_agent()
# ppo_agent(1)
# ppo.act(env)
# ppo.test()
# ppo.act(env)
# ddpg_agent()

#  def random_agent():
#     env.render()
#     action = np.random.uniform(-1.0,1.0,size=action_size)
#     observation, reward, done, info = env.step(action)
#     if done: 
#       observation = env.reset()     

  # modelname = f'Student_1'
  # test_episodes = 100
  # agent = ppo.PPOAgent(env_name,'Student/',modelname)
  # agent.load()
  # for e in range(101):
  #   state = agent.env.reset()
  #   state = np.reshape(state, [1, agent.state_size[0]])
  #   done = False
  #   score = 0
  #   while not done:
  #       agent.env.render()
  #       get_teacher_actions(state)
  #       # CHANGE HERE - INSTEAD OF PREDICTING, GRAB THE POSSIBLE ACTIONS
  #       action = agent.Actor.predict(state)[0]
  #       print(action)
  #       state, reward, done, _ = agent.env.step(action)
  #       state = np.reshape(state, [1, agent.state_size[0]])
  #       score += reward
  #       if done:
  #           average, SAVING = agent.PlotModel(score, e, save=False)
  #           print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
  #           break
  #   agent.env.close()
