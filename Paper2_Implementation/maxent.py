import numpy as np
import gym
from tqdm import tqdm
import math

seed = 1
np.random.seed(seed)

P = None

state_index_mapping = np.load('state_mapping_mc.npy', allow_pickle=True).item()


# print(type(state_index_mapping),state_index_mapping)
# print(state_index_mapping)
def transition_probabilities(i1, action, i2, num_x=20, num_v=20):
    global state_index_mapping
    # print(type(state_index_mapping))
    # print(state_index_mapping[0],"issue here")
    s1 = state_index_mapping[i1]
    s2 = state_index_mapping[i2]
    x_unit = 1.8 / num_x
    v_unit = 0.14 / num_v
    force = 0.001
    gravity = 0.0025
    max_speed = 0.07
    min_position = -1.2
    max_position = 0.6
    goal_position = 0.5
    goal_velocity = 0

    position = x_unit * s1[0] - 1.2
    velocity = v_unit * s1[1] - 0.07

    dv = (action - 1) * force + math.cos(3 * position) * (-gravity)
    velocity += dv
    velocity = np.clip(velocity, -max_speed, max_speed)
    dx = velocity
    position += dx
    position = np.clip(position, min_position, max_position)
    if (position == min_position and velocity < 0): velocity = 0

    done = bool(position >= goal_position and velocity >= goal_velocity)

    new_position = (1.2 + position) // x_unit
    new_velocity = (0.07 + velocity) // v_unit
    if (new_position == s2[0] and new_velocity == s2[1]):
        return (1)
    return (0)


def transition_prob(i, j, k):
    if (P is not None):
        return (P[i, j, k])
    else:
        return (transition_probabilities(i, j, k))


def backward_pass(n_states, n_actions, traj_length, terminal, rewards):
    z_states = np.zeros((n_states))
    z_actions = np.zeros((n_states, n_actions))
    z_states[terminal] = 1
    for n in range(traj_length):
        for i in range(n_states):
            for j in range(n_actions):
                curr_sum = 0
                for k in range(n_states):
                    c1 = transition_prob(i, j, k)
                    assert c1 <= 1
                    # print('c1',c1)
                    c2 = np.exp(rewards[i])
                    # print('c2',c2)
                    c3 = z_states[k]
                    # print('c3',c3)
                    curr_sum += c1 * c2 * c3
                    # if(curr_sum<=0):
                    #     print("{0},{1},{2}".format(c1,c2,c3))
                z_actions[i, j] = curr_sum
            z_states[i] = np.sum(z_actions[i, :])
            z_states[i] = z_states[i] + 1 if i == terminal else z_states[i]

    return (z_states, z_actions)


def local_action_probability_computation(z_states, z_actions):
    policy = np.zeros(z_actions.shape)
    for i in range(z_actions.shape[0]):
        for j in range(z_actions.shape[1]):
            policy[i, j] = z_actions[i, j] / z_states[i] if z_states[i] > 0 else 0
    return (policy)


def forward_pass(policy, trajectories, traj_length):
    D_t = np.zeros((policy.shape[0], traj_length))
    for i in trajectories:
        D_t[i[0][0], :] += 1 / len(trajectories)
    # D_t[:, :] = D_t[:, :] / len(trajectories)

    for s in range(policy.shape[0]):
        for t in range(traj_length - 1):
            D_t[s, t + 1] = sum(
                [sum([D_t[k, t] * policy[k, a] * transition_prob(k, a, s) for k in range(policy.shape[0])]) for a in
                 range(policy.shape[1])])

    D = np.sum(D_t, 1)

    return (D)


def expected_edge_frequency_calculation(trajectories, terminal, rewards, n_states, n_actions):
    traj_length = len(trajectories[0])
    z_s, z_a = backward_pass(n_states, n_actions, traj_length, terminal, rewards)
    policy = local_action_probability_computation(z_s, z_a)
    D = forward_pass(policy, trajectories, traj_length)
    return (D)


def update(theta, alpha, f_expert, f, D):
    gradient = f_expert - np.dot(f.T, D)
    theta += alpha * gradient
    return (theta)


def expert_feature_expectations(trajectories, features):
    exps = np.zeros((features.shape[1]))
    for i in trajectories:
        for s in i:
            # print(s)
            f = features[s[0], :]
            exps += f / len(trajectories)
    return (exps)


def irl(pr, features, trajectories, epochs, alpha, n_states, n_actions, true_rewards):
    global P
    P = pr
    errors = []
    terminal = trajectories[0][-1][0]
    true_rewards = true_rewards / np.linalg.norm(true_rewards)
    theta = np.random.uniform(size=(features.shape[1]))
    exps = expert_feature_expectations(trajectories, features)
    for i in tqdm(range(epochs)):
        rewards = np.dot(features, theta)
        rewards = rewards / np.linalg.norm(rewards) if not np.all(rewards == 0) else rewards
        er = np.linalg.norm(rewards - true_rewards)
        errors.append(er)
        # print("Error at epoch {0}: {1}".format(i,er))
        D = expected_edge_frequency_calculation(trajectories, terminal, rewards, n_states, n_actions)
        theta = update(theta, alpha, exps, features, D)
    # print(rewards)
    rewards = np.dot(features, theta) / np.linalg.norm(rewards)

    np.save('Paper2/rewardsgw', rewards)
    np.save('Paper2/errorsgw', errors)
    return (rewards)
