import gym
env = gym.make("MountainCar-v0")
#env = gym.make("CartPole-v1")
state = env.reset()
import numpy as np
from autograd import grad
import sys

GAMMA = 0.99
ALPHA = 0.1
LAMBDA = 1
NB_STEPS = 5

# (Just debug) Plot the rewards
import matplotlib.pyplot as plt

def q(state, parameters):
    return np.matmul(parameters, state)

# print(q(np.random.random(2), W))
# sys.exit(0)

def loss(parameters, new_state, reward, old_value):
    return ALPHA * (reward + GAMMA * max(q(new_state, parameters)) - old_value)

def run_episode(env):
    parameters = np.random.random((3, 2))
    for _ in range(10):
        print(_)
        state = env.reset()
        # totalreward = 0
        for _ in range(1000):
            # Choose A from S using policy derived from Q
            action = np.argmax(q(state, parameters))
            # print(action)
            
            # Take action A, observe R, S'
            new_state, reward, done, info = env.step(action)
            # print('all', q(state, parameters))
            # print('that', q(state, parameters)[action])
            old_value = q(state, parameters)[action]
            new_value = old_value + ALPHA * (reward + GAMMA * max(q(new_state, parameters)) - old_value)
        
            # Gradient descent parameters
            for _ in range(NB_STEPS):
                # print(action)
                that = q(state, parameters)[action]
                # print(that, new_value, new_value - that)
                # print('p', parameters)
                # print('nabla', grad(lambda parameters: loss(parameters, new_state, reward, old_value))(parameters))
                parameters -= LAMBDA * grad(lambda parameters: loss(parameters, new_state, reward, old_value))(parameters)
            # totalreward += reward
            # Until S is terminal
            if done:
                break
    # return totalreward
    return parameters

parameters = run_episode(env)

# sys.exit(0)

"""
rewards = []
bestparams = None  
bestreward = -float('inf')
for _ in range(500):  
    # parameters = np.random.rand(2) * 2 - 1
    reward = run_episode(env)
    rewards.append(reward)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break
"""

# You do not need
# plt.hist(rewards)
# plt.show()
        
print(env.action_space)
for _ in range(500):
    env.render()
    # action = env.action_space.sample()  # pick a random action
    # print(type(bestparams))
    # print(type(observation))
    # action = 0 if np.dot(bestparams,observation) < 0 else 2
    # action = 2
    action = np.argmax(q(state, parameters))
    
    # action = 1
    print(action)
    state, reward, done, info = env.step(action)

env.close()
