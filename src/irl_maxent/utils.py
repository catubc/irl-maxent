"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
import gridworld as W                       # basic grid-world MDPs
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P  

#
def setup_mdp():
    # create our world
    world = W.IcyGridWorld(size=5, p_slip=0.2)

    # set up the reward function
    reward = np.zeros(world.n_states)
    reward[-1] = 1.0
    reward[8] = 0.65

    # add additoinal tests
    # reward[16] = 0.65
    # reward[17] = 0.65
    # reward[18] = 0.65


    # set up terminal states
    terminal = [24]

    return world, reward, terminal
    

def generate_expert_trajectories(world, reward, terminal):
    n_trajectories = 200         # the number of "expert" trajectories
    discount = 0.9               # discount for constructing an "expert" policy
    weighting = lambda x: x**50  # down-weight less optimal actions
    start = [0]                  # starting states for the expert

    # compute the value-function
    value = S.value_iteration(world.p_transition, reward, discount)
    
    # create our stochastic policy using the value function
    policy = S.stochastic_policy_from_value(world, value, w=weighting)
    
    # a function that executes our stochastic policy by choosing actions according to it
    policy_exec = T.stochastic_policy_adapter(policy)
    
    # generate trajectories
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, start, terminal))
    
    return tjs, policy

#
def plot_reward_and_expert_trajectories(world,
                                        reward,
                                        style,
                                        expert_policy,
                                        trajectories):

    #
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.title.set_text('Original Reward')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_state_values(ax, world, reward, **style)
    fig.colorbar(p, cax=cax)

    #
    ax = fig.add_subplot(122)
    ax.title.set_text('Expert Policy and Trajectories')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    p = P.plot_stochastic_policy(ax, world, expert_policy, **style)
    fig.colorbar(p, cax=cax)

    for t in trajectories:
        P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

    fig.tight_layout()
    plt.show()

