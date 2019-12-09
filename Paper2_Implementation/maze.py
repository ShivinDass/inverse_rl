import sys
import numpy as np
import math
import random
import time
import gym
import matplotlib.pyplot as plt
import gym_maze

IRL_REWARDS = np.load('rewardsgw.npy',allow_pickle=True)
IRL_REWARDS = IRL_REWARDS - np.max(IRL_REWARDS)

def simulate(ax,color,IRL=False):

    # Instantiating the learning related parameters
    rewards = []
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0

    # Render tha maze
    env.render()

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate)

            # execute the action
            obv, reward, done, _ = env.step(action)


            # Observe the result
            state = state_to_bucket(obv)
            s = state[0]*10+state[1]
            if(IRL):
                reward =IRL_REWARDS[s]
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE and num_streaks==STREAK_TO_END:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        rewards.append(total_reward)
    l = 'IRL Rewards' if IRL else 'True Rewards'
    ax.plot(rewards,color=color)
    # plt.show()



def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":

    # Initialize the "maze" environment
    env = gym.make("maze-sample-10x10-v0")

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 30.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 50000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = False
    ENABLE_RECORDING = False

    '''
    Creating a Q-Table for each state-action pairq
    '''
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    '''
    Begin simulation
    '''

    recording_folder = "/"

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episodic True Reward', color=color)

    simulate(ax1,color)
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Episodic IRL Reward', color=color)  # we already handled the x-label with ax1
    simulate(ax2,color,True)
    # ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()


    # plt.xlim([-5,5])
    # plt.ylim([-5,5])
    plt.legend()
    plt.show()

    if ENABLE_RECORDING:
        env.monitor.close()
    trajectories = []
    for episode in range(100):
        curr_traj = []
        # Reset the environment
        obv = env.reset()

        # the initial state
        state = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state, 0)
            s = state[0]*10+state[1]
            curr_traj.append((s,action))
            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            if(done):
                print("done",t)
                while(len(curr_traj)<100):
                    s = state[0]*10+state[1]
                    curr_traj.append((s,action))
                trajectories.append(curr_traj)
                break
            # env.render()
            # time.sleep(1)

    # transition_matrix = np.zeros((100,4,100))
    # for i in range(10):
    #     for j in range(10):
    #         for a in range(4):
    #             for k in range(10):
    #                 for l in range(10):
    #                     env.reset()
    #                     s1 = np.array([i,j],dtype=int)
    #                     s2 = np.array([k,l],dtype=int)
    #                     id1 = i*10+j
    #                     id2 = k*10+l
    #                     env.__robot=s1
    #                     env.step(a)
    #                     p = 1 if (env.__robot[0]==s2[0] and env.__robot[1]==s2[1]) else 0
    #                     transition_matrix[id1,a,id2]=p
    #
    # np.save('maze_transition_matrix',transition_matrix)
    # np.save('qtable_maze',q_table)
    # np.save('50_maze_traj',trajectories)