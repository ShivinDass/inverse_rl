import numpy as np
import matplotlib.pyplot as plt

from Paper2 import maxent


def feature_matrix(states=120 * 120):
    features = []
    for i in range(states):
        f = np.zeros(states)
        f[i] = 1
        features.append(f)
    return np.array(features)

def main(grid_size, discount, n_trajectories, epochs, learning_rate):

    trajectory_length = 80
    n_states = 100
    n_actions = 4
    trajectories = np.load('Paper2/50_maze_traj.npy',allow_pickle=True)

    fm = feature_matrix(100)
    ground_r = np.array([-1/n_states for s in range(n_states)])
    ground_r[-1]=1
    transition_probability = np.load('Paper2/maze_transition_matrix.npy',allow_pickle=True)
    r = maxent.irl(transition_probability, fm, trajectories, epochs, learning_rate, n_states, n_actions, ground_r)
    print(r)
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    # plt.subplot(1, 3, 3)
    # plt.pcolor(r1.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Recovered reward")


    plt.show()

if __name__ == '__main__':
    main(10, 0.01, 100, 100, 0.01)