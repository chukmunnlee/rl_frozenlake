#import libs
import time
import numpy as np
# openai gym
import gym
from matplotlib import pyplot as plt
from tqdm import tqdm
from print_policy import print_policy

# create the environment with the name
env = gym.make('FrozenLake-v0')
MAP_SIZE = 4 # 4 x 4

#env = gym.make('FrozenLake8x8-v0')
#MAP_SIZE = 8 # 4 x 4
#env = gym.make('CartPole-v0')

ACTION_SPACE = 4 # N, S, E, W
# as episode progress - so does my random action
EPSILON_SCHEDULE = [ .9, .8, .7, .6, .5, .4, .3, .2, .1, .05, .01 ]
EPSILON_SCHEDULE = [ .9, .9, .9, .9, .5, .5, .2, .2, .01, .01, .01 ]
K = 1000

EPISODES = K * len(EPSILON_SCHEDULE)

q_table = np.zeros((MAP_SIZE * MAP_SIZE, ACTION_SPACE))

state_action_reward = np.zeros((MAP_SIZE * MAP_SIZE, ACTION_SPACE))
state_action_count = np.zeros((MAP_SIZE * MAP_SIZE, ACTION_SPACE))

q_table = np.zeros((MAP_SIZE * MAP_SIZE, ACTION_SPACE))

state_action_reward = np.zeros((MAP_SIZE * MAP_SIZE, ACTION_SPACE))
state_action_count = np.zeros((MAP_SIZE * MAP_SIZE, ACTION_SPACE))

# randomness to action selection
def epislon(ep):
   return EPSILON_SCHEDULE[ ep // K ]

# epsilon-greedy policy
def q(state, e):
   # with 1 - epsilon probability, pick the greedy action
   if np.random.uniform() > e:
      # take the greedy action
      return np.argmax(q_table[state])
   # with epsilon probability, pick a random action including the greedy action
   return np.random.randint(0, ACTION_SPACE)

success = []
success_rate = []

for ep in tqdm(range(EPISODES)):

   # start the environment - every when an episode ends
   state = env.reset()
   done = False

   # render the environment
   # env.render()

   rollouts = []
   while not done:

      # 4 actions - W (0), S, E, N (3)
      action = q(state, epislon(ep))

      # take the action
      # new_state - result of the action
      # reward - immediate reward for taking the action
      # done - boolean - False - the game has not ended, 
      # True - reached the G, or landed on H - the episode ends
      new_state, reward, done, _ = env.step(action)
      #env.render()

      if done:
         # 0 - falling into the H, 1 - for reaching G
         success.append(reward)
         success_rate.append(sum(success))

      # (state, action, reward)
      rollouts.append((state, action, reward))

      state = new_state

   # completed the episode
   # calculate the average cumulative reward
   visited = set()
   for i, v in enumerate(rollouts):
      st, ac, _ = v

      # first visit - have we encountered this (st, ac) pair in the current episode
      if (st, ac) in visited:
         continue

      visited.add((st, ac))
      #count the number of times st,ac has been taken
      state_action_count[st, ac] += 1
      # sum the cumulative reward
      state_action_reward[st, ac] += sum([ r[2] for r in rollouts[i:] ])
      # average cumulative reward
      q_table[st, ac] = state_action_reward[st, ac] / state_action_count[st, ac]

#print('reward')
#print(state_action_reward)
#print('count')
#print(state_action_count)
print('q_table')
print(q_table)

env.render()

print_policy(q_table, MAP_SIZE)

print()

fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot(range(len(success)), success)

ax = fig.add_subplot(122)
ax.plot(range(len(success_rate)), success_rate)

fig.suptitle('Episodes: %d, success: %d, ratio: %.2f' %(EPISODES, success_rate[-1], success_rate[-1]/EPISODES))

plt.show();
