import numpy as np
import matplotlib.pyplot as plt
import gym

# create the environment
env = gym.make('FrozenLake-v0')

# everytime when the game ends, we need to reset the environement
state = env.reset()

print("Resetted environment rendered map")
env.render()

# Take a random action out of 4 actions - W(0), S, E, N (3)
action = np.random.randint(0,4)

# After taking an action, the following will be returned
# new_state - result of action
# reward - immediate reward for taking the action
# done - boolean of whether the game has ended
new_state, reward, done, _ = env.step(action)

print("========")
print("New State: ", new_state)
print("Reward: ", reward)
print("Done: ", done)
print("Probability: ", _)
print("========")

print("Action taken and new state in rendered map:")
env.render()

