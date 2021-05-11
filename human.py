import gym
env = gym.make("BipedalWalker-v3")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print(f'observation: {observation}')
  print(f'reward: {reward}')
  print(f'done: {done}')
  if done:
    print("done")
    observation = env.reset()
env.close()