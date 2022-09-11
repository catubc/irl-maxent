"""
Trajectories representing expert demonstrations and automated generation
thereof.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

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

#
def plot_trajectory_simulated(traj,n_col,n_row):

    #
    #print ("plotting trajectory: ", traj)

    # compute locations
    loc_ids = np.arange(n_col*n_row)
    loc_xy = np.zeros((loc_ids.shape[0],2))
    for k in range(loc_ids.shape[0]):
        #print (loc_xy[k], )
        loc_xy[k] = [loc_ids[k]%n_col, loc_ids[k]//n_col]
        #print (k, loc_xy[k])

    #
    colors = plt.cm.jet(np.linspace(0,1,traj.shape[0]))

    # make grid world:
    for k in range(n_row+1):
        plt.plot([0,n_col],[k,k],
                 c='black',
                 alpha=0.2)
        #break
    for k in range(n_col+1):
        plt.plot([k,k],[0,n_row],
                 c='black',
                 alpha=0.2)
    #
    for k in range(n_col*n_row):
        if k%100==0:
            plt.text(k%n_col+0.5, k//n_col+0.5, str(k),
                     fontsize=6)


    print ("plotted traj: ", np.vstack(traj))
    for k in range(traj.shape[0]):
        #t = traj[k]
        s = traj[k][0]
        e = traj[k][2]

        #
        #print ("k: ", k, ", s, e: ", s,e, "s xy: ", loc_xy[s], " e xy: ", loc_xy[e])
        temp = np.vstack((loc_xy[s], loc_xy[e]))
        plt.plot(temp[:,0]+0.5, temp[:,1]+0.5,
                  c=colors[k])


#
def plot_trajectory_gerbils(traj,
                    n_col,
                    n_row,
                    grid_states,
                    plot_text=True):

    #
    colors = plt.cm.jet(np.linspace(0,1,traj.shape[0]))

    # make grid world:
    for k in range(n_row+2):
        plt.plot([0,n_col-1],
                 [k-1,k-1],
                 c='black',
                 alpha=0.2)
        #break
    for k in range(n_col):
        plt.plot([k,k],
                 [-1,n_row],
                 c='black',
                 alpha=0.2)
    #
    if plot_text:
        for k in range(n_col*n_row):
            x = k//n_col+0.5
            y = n_row-(k)%n_col-0.5
            #print ("x, y: ", x,y, k//n_row,-(k)//n_col+n_row)
            plt.text(x,
                     y,
                     str(k),

                     fontsize=6)


    #print ("plotted traj locations: ", np.vstack(traj))
    noise_old = np.random.rand(2) / 5. - 0.2
    for k in range(traj.shape[0]):
        #t = traj[k]
        s = traj[k][0]
        e = traj[k][2]

        #
        noise_new = np.random.rand(2) / 5. - 0.2
        # print ("k: ", k, ", s, e: ", s,e, "s xy: ", loc_xy[s], " e xy: ", loc_xy[e])

        #
        idx_s = np.where(grid_states==s)
        idx_e = np.where(grid_states==e)

        #print ("Idx s: ", idx_s,
        #       " idx_e,: ", idx_e)
        temp = np.vstack((np.hstack(idx_s)[::-1]+noise_old,
                          np.hstack(idx_e)[::-1]+noise_new))
        noise_old = noise_new

        #
        plt.plot(temp[:,0]+0.5,
                 n_row-temp[:,1]-0.5,
                  c=colors[k])

    #


# @jit
def fix_trajectories(track_xy1,
                     max_jump_allowed=100,
                     max_dist_to_join=100,
                     min_chunk_len=5,
                     plotting=False):
    ''' Method to fix the large jumps, short orphaned segments,
        and interpolate across short distances

        Input:
        - track array for a single animal: [n_time_steps, 2], where the 2 is for x-y locations
        - max_jump_allowed: maximum number of pixels (in euclidean distance) a track is allowed to jump before being split
        - max_dist_to_join: when joining orphaned tracks, the largest distacne allowed between 2 track bouts
        - min_chunk_len = shortest chunk allowed (this is applied at the end after joining all the chunks back

        Output: fixed track

    '''

    ########################################
    ###########  Delete big jumps ##########
    ########################################
    for k in trange(1, track_xy1.shape[0] - 1, 1):
        if np.linalg.norm(track_xy1[k] - track_xy1[k - 1]) > max_jump_allowed:
            track_xy1[k] = np.nan

    ########################################
    ##### Join segments that are close #####
    ########################################
    #
    last_chunk_xy = None

    # check if we start outside chunk or inside
    if np.isnan(track_xy1[0, 0]) == False:
        inside = True
    else:
        inside = False

    # interpolate between small bits
    for k in trange(1, track_xy1.shape[0] - 1, 1):
        if np.isnan(track_xy1[k, 0]):
            if inside:
                inside = False
                last_chunk_xy = track_xy1[k]
                last_chunk_idx = k
        else:
            if inside == False:
                inside = True
                new_chunk_xy = track_xy1[k]
                new_chunk_idx = k
                if last_chunk_xy is not None:
                    dist = np.linalg.norm(track_xy1[k] - track_xy1[k - 1])
                    if dist <= max_dist_to_join:
                        track_xy1[last_chunk_idx:new_chunk_idx] = new_chunk_xy

    ########################################
    ##  Delete short segments left behind ##
    ########################################
    #

    chunk_start_xy = None

    # check if we start outside chunk or inside
    if np.isnan(track_xy1[0, 0]) == False:
        chunk_start_idx = 0
        inside = True
    else:
        inside = False

    # interpolate between small bits
    for k in trange(1, track_xy1.shape[0] - 1, 1):
        if np.isnan(track_xy1[k, 0]):
            if inside:
                inside = False
                chunk_end_idx = k
                if (chunk_end_idx - chunk_start_idx) < min_chunk_len:
                    track_xy1[chunk_start_idx:chunk_end_idx] = np.nan
        else:
            if inside == False:
                inside = True
                chunk_start_idx = k

    if plotting:
        fig = plt.figure()
        plt.plot(track_xy1[:, 0],
                 track_xy1[:, 1])
        plt.show()

    return track_xy1



#
def compute_trajectories_from_data(track,
                                   box_width,
                                   box_height,
                                   remove_no_movement_states_flag=True,
                                   max_len_trajectory=20,
                                   grid_width=100,
                                   min_len_trajectory=10):

    #
    verbose = False

    #
    n_col = box_width // grid_width
    n_row = box_height // grid_width

    # make a 2d array holding all the state locations
    n_states = n_row * n_col
    print(" n states: ", n_states, ' n_col ', n_col, " n_row ", n_row)
    grid_states = np.arange(n_states, dtype='int32')
    grid_states = np.array_split(grid_states, n_row)
    grid_states = np.vstack(grid_states).T  # [::-1].T
    print("grid states: ", grid_states)

    # convert all trajectories to discrete grid world xy values
    track_xy = track.copy()
    track_xy1 = track_xy[:, 1]

    ################################################
    ############ FIX TRACK ERRRORS #################
    ################################################
    max_jump_allowed = np.sqrt(2 * grid_width ** 2)
    max_dist_to_join = np.sqrt(2 * grid_width ** 2)
    # min_chunk_len=5,

    track_xy1 = fix_trajectories(track_xy1,
                                 max_jump_allowed,
                                 max_dist_to_join)

    #
    track_xy1 = np.float32(track_xy1 // grid_width)

    #
    track_states1 = np.int32(track_xy1[:, 0] + track_xy1[:, 1] * n_col)

    # list to save states of each animal
    states1 = []
    actions_xy1 = []

    # get first state for first animal
    s = track_states1[0]
    e = track_states1[1]
    s_xy = track_xy1[0]  # start xy
    e_xy = track_xy1[1]  # end xy

    #
    active_trajectory = []
    active_trajectory = update_trajectory(active_trajectory,
                                          s_xy,
                                          e_xy,
                                          s,
                                          e,
                                          grid_states)
    # delete the state
    track_states1 = np.delete(track_states1, 0, axis=0)
    track_xy1 = np.delete(track_xy1, 0, axis=0)

    # loop over remaining states
    ctr = 0
    while track_states1.shape[0] > 1:

        # get first animal locations
        s = track_states1[0]
        e = track_states1[1]
        s_xy = track_xy1[0]  # start xy
        e_xy = track_xy1[1]  # end xy

        # do a quick check to make sure start and end are nearby:
        delta_x = np.abs(s_xy[0] - e_xy[0])
        delta_y = np.abs(s_xy[1] - e_xy[1])
        if (delta_x + delta_y) <= 2 and np.any(np.isnan(s_xy)) == False and np.any(np.isnan(e_xy)) == False:

            # if len(states1)==2:
            # print ("delta x, y:", delta_x, delta_y,
            #       "sxy: ", s_xy, e_xy,
            #       's, e: ', s,e,
            #       ", active_trajectory: ", active_trajectory)
            active_trajectory = update_trajectory(active_trajectory,
                                                  s_xy,
                                                  e_xy,
                                                  s,
                                                  e,
                                                  grid_states)

            # TODO: note this conditional essentially removes all states where the gerbils do not do anything;
            # if remove_no_movement_states_flag:
            #    if a!=4:
            #        #print ("up-to-date active: ", active_trajectory)
            #        active_trajectory.append([s,a,e])
            # else:
            #    active_trajectory.append([s,a,e])

            if len(active_trajectory) == max_len_trajectory:
                states1.append(active_trajectory)
                if verbose:
                    print("completed trajectory :", len(states1) - 1)
                    print("trajectory: ", active_trajectory)
                    print("##########################################################")
                    print("")
                    print("")
                # print (ctr, "appended trajectroy: ", len(active_trajectory),
                #       active_trajectory)
                active_trajectory = []
        else:
            if len(active_trajectory) >= min_len_trajectory:
                states1.append(active_trajectory)
                if verbose:
                    print("completed trajectory :", len(states1) - 1)
                    print("trajectory: ", active_trajectory)
                    print("############ TRAJECTORY " + str(len(states1) - 1) + "######################")
                    print("")
                    print("")
                # print (ctr, "appended trajectroy: ", len(active_trajectory),
                #       active_trajectory)
            active_trajectory = []

        #
        # delete current value
        track_states1 = np.delete(track_states1, 0, axis=0)
        track_xy1 = np.delete(track_xy1, 0, axis=0)

        #
        ctr += 1

    return states1, actions_xy1, n_col, n_row, grid_states



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


#


#
def get_action_xy_shifts(a):
    # compute actions
    # actions = [0, 1, 2, 3, 4]
    # action_names = ['right', 'left', 'up', 'down', 'nothing']

    if a == 0:
        shifts = [+1, 0]
    elif a == 1:
        shifts = [-1, 0]
    elif a == 2:
        shifts = [0, +1]
    elif a == 3:
        shifts = [0, -1]
    elif a == 0:
        shifts = [0, 0]
    else:
        print("ERROR computeing direction:")
        errror

    return np.int32(shifts)

#
def update_trajectory(active_trajectory,
                      start_xy,
                      end_xy,
                      start_state,
                      end_state,
                      grid_states):

    # compute actions
    actions = [0, 1, 2, 3, 4]
    action_names = ['right', 'left', 'up', 'down', 'nothing']

    # compute actions
    if (start_xy[0] - end_xy[0]) == 0 and (start_xy[1] - end_xy[1]) == 0:
        action = [4]
    elif ((start_xy[0] - end_xy[0]) == -1) and ((start_xy[1] - end_xy[1]) == 0):
        action = [1]
    elif ((start_xy[0] - end_xy[0]) == +1) and ((start_xy[1] - end_xy[1]) == 0):
        action = [0]
    elif ((start_xy[1] - end_xy[1]) == -1) and ((start_xy[0] - end_xy[0]) == 0):
        action = [3]
    elif ((start_xy[1] - end_xy[1]) == +1) and ((start_xy[0] - end_xy[0]) == 0):
        action = [2]
    else:
        # DIAGONAL movement
        if (start_xy[0] - end_xy[0]) == -1:  # right
            if (start_xy[1] - end_xy[1]) == -1:
                action = [0, 2]  # up
            else:
                action = [0, 3]
        else:  # left
            if (start_xy[1] - end_xy[1]) == -1:
                action = [1, 2]  # up
            else:
                action = [1, 3]

        # shuffle the order of the actions because we don't really know which way the animal went
        # e.g. down, right vs. right, down
        action = np.random.choice(action, 2, replace=False)

   # print ("action detected: ", action)


    # update trajectory
    if len(action) == 1:
        if action[0] != 4:
            transition = [start_state,
                          action[0],
                          end_state]
            active_trajectory.append(transition)
            #print ("detected regular movement", transition)

    #
    elif len(action) == 2:

        # select starting state and action for the first action
        start_state0 = start_state  # this is the starting state

        # here we find which xy directio the action takes us
        # return a tuple -1,0,1 in each dimension
        shift = get_action_xy_shifts(action[0])
        new_x = int(start_xy[0]) + shift[0]
        new_y = int(start_xy[1]) + shift[1]

        #
        if False:
            print ("Diagonal, first step: ", shift,
               " start_xy: ", start_xy,
               " end_xy: ", end_xy,
               " start_state0: ", start_state0,
               " end_states: ", end_state,
               " new_x: ", new_x,
               " new_y: ", new_y
                   , "grid states shaep: ", grid_states.shape
               )
        # here we compute the end state by moving across the grid_states 2d array
        end_state0 = grid_states[new_x,
                                 new_y,
                                 ]
        # print ("new end state: ", end_state0)
        # print ("Grid states: ", grid_states, )

        #
        transition = [start_state0,
                      action[0],
                      end_state0]
        # print ("Appending transition #1: ", transition)
        #print (" second step")

        #
        #print ("detected diagonal movment, transiition 1 ", transition)
        active_trajectory.append(transition)

        # select starting state and action for the second action
        start_state1 = end_state0
        end_state1 = end_state  # this is the end state that we expect to end up in

        #
        transition = [start_state1,
                      action[1],
                      end_state1]
        # print("Appending transition #2: ", transition)
        # print ("detected diagonal movment, transiition 2 ", transition)
        active_trajectory.append(transition)

    else:
        print("ERRROr in action computation: ")
        error

    return active_trajectory
