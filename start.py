"""
Episodic semi-gradient Sarsa for Gym's MountainCar v0
JJ Vie, 2020
"""
# pylint: disable=redefined-outer-name,import-error,
# pylint: disable=pointless-string-statement,no-member,unused-variable
import random
import gym
import numpy as np
import matplotlib.pyplot as plt  # For plotting actions
from tiles3 import IHT, tiles
# from autograd import grad

env = gym.make("MountainCar-v0")
# env = gym.make("CartPole-v1")
actions = range(3)
state = env.reset()

GAMMA = 0.99
ALPHA = 0.1
EPS = 0.9
NB_EPISODES = 100
TIMEOUT = 3000

iht = IHT(4096)


def q(state, action, parameters):
    """
    Our function approximation for q(s, a)
    """
    return parameters[embed(state, action)].sum()


def embed(state, action):
    """
    States are embedded in {0, 1}^4096 using tile coding
    See Richard S. Sutton's http://incompleteideas.net/tiles/tiles3.html
    """
    return tiles(iht, 8, [8 * (state[0] / (0.5 + 1.2)),
                          8 * (state[1] / (0.07 + 0.07))], [action])


def run_episode(env):
    """
    Run several episodes of sarsa
    """
    parameters = np.random.random(4096) * 2 - 1
    all_states = []
    length = {x: [] for x in range(NB_EPISODES)}
    for n in range(NB_EPISODES):
        if n:
            print(n, np.mean(length[n - 1]))
        state = env.reset()
        sarsa = [state]

        # Choose A from S using eps-greedy policy derived from Q
        action = (np.argmax([q(state, action, parameters)
                             for action in actions]) if random.random() < EPS
                  else random.choice(actions))
        # totalreward = 0
        t = 0
        while True:
            if n == NB_EPISODES - 1:  # If final episode, let's show it
                env.render()

            # Take action A, observe R, S'
            new_state, reward, done, info = env.step(action)
            all_states.append(new_state)

            # If state S' is terminal
            if done:
                length[n].append(len(sarsa))
                parameters[embed(state, action)] += ALPHA * (
                    reward - q(state, action, parameters))
                break  # Go to next episode

            # Choose A' as a eps-greedy function of q(S', .)
            new_action = (
                np.argmax([q(new_state, action, parameters)
                           for action in actions]) if random.random() < EPS
                else random.choice(actions))

            # If we want to remember everything
            # sarsa.extend((action, reward, new_state))
            sarsa.append(new_state)

            parameters[embed(state, action)] += ALPHA * (
                reward + GAMMA * q(new_state, new_action, parameters) -
                q(state, action, parameters))

            if t == TIMEOUT:
                # print(sarsa)
                length[n].append(len(sarsa))
                break

            state = new_state
            action = new_action

            t += 1

            """
            # That was q-learning (off-policy)
            old_value = q(state, parameters)[action]
            new_value = (old_value + ALPHA *
                         (reward + GAMMA * max(q(new_state, parameters)) -
                          old_value))

            # Gradient descent
            for _ in range(NB_STEPS):
                parameters -= LAMBDA * grad(loss)(parameters)
            totalreward += reward
            # Until S is terminal
            if done:
                break"""
    # return totalreward
    return parameters, all_states


# Training
parameters, all_states = run_episode(env)
all_ = np.array(list(all_states))
print(all_.min(axis=0), all_.max(axis=0))  # Some stats about the states

# Plot the best action
points = {action: [[], []] for action in actions}
for x in np.linspace(-1.2, 0.5, 50):
    for y in np.linspace(-0.08, 0.08, 50):
        best_action = np.argmax([q([x, y], action, parameters)
                                 for action in actions])
        points[best_action][0].append(x)
        points[best_action][1].append(y)

plt.title('Best action')
for action in points:
    plt.scatter(points[action][0], points[action][1],
                c='rgb'[action], label=['left', 'stop', 'right'][action])
plt.legend()
plt.show()

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
            break"""


state = env.reset()
for _ in range(500):
    env.render()
    # Random
    # action = env.action_space.sample()  # pick a random action

    # Dot product
    # action = 0 if np.dot(bestparams, observation) < 0 else 2

    # Greedy
    # action = np.argmax(q(state, action, parameters) for action in range(3))

    # Epsilon greedy
    action = (np.argmax([q(state, action, parameters)
                         for action in actions]) if random.random() < EPS
              else random.choice(actions))

    state, reward, done, info = env.step(action)
    if done:
        print('hooray', info)
        break

env.close()
