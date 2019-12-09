import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

from Paper2 import maxent, gridworld


def main(grid_size, discount, n_trajectories, epochs, learning_rate):

    wind = 0.3
    trajectory_length = 7*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)
    trajectories = gw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            gw.optimal_policy)
    feature_matrix = gw.feature_matrix()
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    r = maxent.irl(gw.transition_probability, feature_matrix, trajectories, epochs, learning_rate, gw.n_states, gw.n_actions, ground_r)

    # plt.subplot(1, 2, 1)
    # plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Ground Truth reward")
    # plt.subplot(1, 2, 2)
    # plt.pcolor(r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("Recovered reward")


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 5, 1)
    X, Y = np.meshgrid(x, y)
    zs = r
    Z = zs.reshape(X.shape)
    ax.view_init(45, 135)
    ax.plot_surface(X, Y, Z,alpha=0.5,cmap='jet', rstride=1, cstride=1, edgecolors='k', lw=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Reward Values')

    plt.show()

if __name__ == '__main__':
    main(5, 0.01, 100, 20, 0.01)