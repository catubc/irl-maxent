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
                                        #style,
                                        expert_policy,
                                        trajectories):
    style = {                                   # global style for plots
        'border': {'color': 'red', 'linewidth': 0.5},
    }

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

def plot_trajectory(traj):

    # compute locations
    loc_ids = np.arange(25)
    loc_xy = np.zeros((loc_ids.shape[0],2))
    for k in range(loc_ids.shape[0]):
        #print (loc_xy[k], )
        loc_xy[k] = [loc_ids[k]%5, loc_ids[k]//5]
        #print (k, loc_xy[k])

    #
    colors = plt.cm.jet(np.linspace(0,1,traj.shape[0]))

    # make grid world:
    for k in range(6):
        plt.plot([0,5],[k,k],
                 c='black')
        #break
    for k in range(6):
        plt.plot([k,k],[0,5],
                 c='black')

    for k in range(traj.shape[0]):
        t = traj[k]
        s = traj[k][0]
        e = traj[k][2]

        #
        #print ("k: ", k, ", s, e: ", s,e, "s xy: ", loc_xy[s], " e xy: ", loc_xy[e])
        temp = np.vstack((loc_xy[s], loc_xy[e]))
        plt.plot(temp[:,0]+0.5, temp[:,1]+0.5,
                  c=colors[k])
    #


##################################################
# # Let's start with the easy parts:
# Remember that the gradient we need to compute for optimization is
#
# $$
#     \nabla_\omega \mathcal{L}(\omega)
#     = \underbrace{\mathbb{E}_{\pi^E} \left[ \phi(\tau) \right]}_{(1)}
#         - \underbrace{\sum_{s_i} D_{s_i} \phi(s_i)}_{(2)},
# $$
#
# so we first need to get the feature expectation $\mathbb{E}_{\pi^E} \left[ \phi(\tau) \right]$ (part (1)) from the trajectories of our trainings-set $\mathcal{D}$.
# This is fairly simple, as for each trajectory we just need to check which states are going to be visited, sum the features of these states up (counting repetitions), and then at the end average over all trajectories, i.e.
#
# $$
#     \mathbb{E}_{\pi^E} \left[ \phi(\tau) \right]
#     = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{s_t \in \tau} \phi(s_t)
# $$
#
# or as a function:
def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s in t.states():                # for each state in trajectory
            fe += features[s, :]            # sum-up features

    return fe / len(trajectories)           # average over trajectories

#########################################################
# Another fairly simple thing we need to know about the trajectories is the probability of a state being a starting state, $p(s_0)$.
# This is required for the state-visitation-frequency $D_{s_i}$ which we in turn need for part (2) of the gradient.
# Here, we just count the number of trajectories starting in each state and then normalize with the number of trajectories.
def initial_probabilities_from_trajectories(n_states, trajectories):
    p = np.zeros(n_states)

    for t in trajectories:                  # for each trajectory
        p[t.transitions()[0][0]] += 1.0     # increment starting state

    return p / len(trajectories)            # normalize
