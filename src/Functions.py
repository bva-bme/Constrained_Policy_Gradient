import torch
import matplotlib.pyplot as plt
import numpy as np
from qpsolvers import solve_qp


# Plot the average scores
def plot_data(data, win):
    ax = plt.axes()
    x = torch.linspace(0, len(data[0]), len(data[0]))
    win = [win] * len(x)
    ax.plot(x, data[0])
    ax.plot(x, win, color='black')
    ax.fill_between(x, data[0] + data[1], data[0] - data[1], alpha=0.2)

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Score')
    plt.show()


# Compute mean and std. of the learning
def compute_learning_stats(scores):
    mean_average_scores = torch.mean(torch.tensor(scores), 0)  # Remove outliers
    std_average_scores = torch.std(torch.tensor(scores), 0)

    return [mean_average_scores, std_average_scores]


# Select an action based on the actual policy
def select_action(agent, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = agent(state)  # Output of the NN (probabilities due to softmax nonlinearity
    m = torch.distributions.Categorical(probs)  # Create a distribution with probabilities from the agent's output
    action = m.sample()  # Sample one (index) from the policy
    log_prob = m.log_prob(action)
    return action.item(), log_prob


# G_k = \sum_{\kappa =k+1}^{n_B} \gamma^{\kappa-k}r_\kappa
def compute_gain(reward_batch, discount_factor=1):
    batch_len = len(reward_batch)
    gain_batch = [0] * batch_len
    for k in range(batch_len, 0, -1):
        k_down = k-1
        if k == batch_len:
            gain_batch[k_down] = reward_batch[k_down]
        else:
            gain_batch[k_down] = reward_batch[k_down] + discount_factor * gain_batch[k_down+1]
    return torch.tensor(gain_batch, dtype=torch.float32)


# Compute log probabilities for the safe states
def compute_logprobs(agent, safe_state_batch, safe_action_batch):
    logprobs = []
    for i in range(safe_state_batch.shape[0]):
        state = safe_state_batch[i].unsqueeze(0)
        logprobs.append(torch.log(agent(state)[0][safe_action_batch[i]].unsqueeze(0)))
    return logprobs


# Compute how much the policy needs to change at the safe states (contains both actions)
def compute_delta_pi_for_safety(agent, safe_state_batch, safe_action_batch, num_safe_states, safety_thd=0.95):
    des_policy_change = torch.zeros([num_safe_states])
    for i in range(num_safe_states):
        state = safe_state_batch[i].unsqueeze(0)
        des_policy_change[i] = safety_thd - agent(state)[0][safe_action_batch[i]].item()
    return des_policy_change


# Compute the gains for the equality constraints
# d pi / dt = theta * gamma * 1/pi
def compute_policy_evolution_safety_eq(agent, state_batch, action_batch, gain_batch, safe_state_batch,
                                       safe_action_batch, device):
    num_safe_states = len(safe_state_batch)
    dpi_des = compute_delta_pi_for_safety(agent, safe_state_batch, safe_action_batch, num_safe_states)
    theta_safe = agent.compute_neural_tangent_kernel(torch.cat([state_batch, safe_state_batch])).to(device)  
    theta_safe_lower_left_block_0 = theta_safe[state_batch.shape[0] * agent.output_size:theta_safe.shape[0],
                                               0:state_batch.shape[0] * agent.output_size]
    theta_safe_lower_right_block_0 = theta_safe[state_batch.shape[0] * agent.output_size:theta_safe.shape[0],
                                                state_batch.shape[0] * agent.output_size:theta_safe.shape[1]]

    theta_safe_lower_left_block = torch.zeros([num_safe_states, theta_safe_lower_left_block_0.shape[1]],
                                              dtype=torch.float64)
    theta_safe_lower_right_block = torch.zeros([num_safe_states, theta_safe_lower_right_block_0.shape[1]],
                                               dtype=torch.float64)
    for i in range(num_safe_states):
        theta_safe_lower_left_block[i, :] = theta_safe_lower_left_block_0[agent.output_size*i+safe_action_batch[i], :]
        theta_safe_lower_right_block[i, :] = theta_safe_lower_right_block_0[agent.output_size*i+safe_action_batch[i], :]

    gains_norm = (gain_batch - gain_batch.mean()) / (gain_batch.std() + 1E-8)
    gamma_known = torch.zeros([len(action_batch)*agent.output_size, len(action_batch)*agent.output_size],
                              dtype=torch.float64)
    for i in range(len(action_batch)):
        gamma_block = torch.zeros([agent.output_size, agent.output_size])
        a = action_batch[i]  # get the index of the action at that step
        gamma_block[a, a] = gains_norm[i]
        gamma_known[i * agent.output_size:i * agent.output_size + agent.output_size,
                    i * agent.output_size:i * agent.output_size + agent.output_size] = gamma_block

    pi_inv_known = torch.zeros([len(action_batch) * agent.output_size], dtype=torch.float64)
    for i in range(len(state_batch)):
        probs = agent(state_batch[i].unsqueeze(0))
        invprobs = 1 / (probs + 1E-8)
        pi_inv_known[i * agent.output_size:i * agent.output_size + agent.output_size] = invprobs

    pi_inv_safety_diag = torch.zeros([num_safe_states * agent.output_size, num_safe_states], dtype=torch.float64)
    for i in range(num_safe_states):
        probs = agent(safe_state_batch[i].unsqueeze(0))
        invprobs = 1 / (probs + 1E-8)
        a = safe_action_batch[i]  # get the index of the action at that step
        pi_inv_safety_diag[i*agent.output_size+a, i] = invprobs[0][a]

    # Linear equation system: Ax = B
    B = (dpi_des - torch.mv(torch.matmul(theta_safe_lower_left_block, gamma_known),
                            pi_inv_known)).unsqueeze(0).transpose(0, 1)
    A = torch.matmul(theta_safe_lower_right_block, pi_inv_safety_diag)

    safety_gains = torch.linalg.solve(A, B)
    learn_gains = torch.cat([gains_norm, safety_gains.squeeze()])

    return learn_gains


# Compute the gains for the inequality constraints
def compute_policy_evolution_safety_ineq(agent, state_batch, action_batch, gain_batch, safe_state_batch,
                                         safe_action_batch, learning_rate, device):
    # d pi / dt = theta * gamma * 1/pi
    num_safe_states = len(safe_state_batch)
    dpi_des = compute_delta_pi_for_safety(agent, safe_state_batch, safe_action_batch, num_safe_states)
    theta_safe = agent.compute_neural_tangent_kernel(torch.cat([state_batch, safe_state_batch])).to(device)
    theta_safe_lower_left_block_0 = theta_safe[state_batch.shape[0] * agent.output_size:theta_safe.shape[0],
                                               0:state_batch.shape[0] * agent.output_size]
    theta_safe_lower_right_block_0 = theta_safe[state_batch.shape[0] * agent.output_size:theta_safe.shape[0],
                                                state_batch.shape[0] * agent.output_size:theta_safe.shape[1]]

    theta_safe_lower_left_block = torch.zeros([num_safe_states, theta_safe_lower_left_block_0.shape[1]],
                                              dtype=torch.float64)
    theta_safe_lower_right_block = torch.zeros([num_safe_states, theta_safe_lower_right_block_0.shape[1]],
                                               dtype=torch.float64)
    for i in range(num_safe_states):
        theta_safe_lower_left_block[i, :] = theta_safe_lower_left_block_0[agent.output_size*i+safe_action_batch[i], :]
        theta_safe_lower_right_block[i, :] = theta_safe_lower_right_block_0[agent.output_size*i+safe_action_batch[i], :]

    gains_norm = (gain_batch - gain_batch.mean()) / (gain_batch.std() + 1E-8)
    gamma_known = torch.zeros([len(action_batch) * agent.output_size, len(action_batch) * agent.output_size],
                              dtype=torch.float64)
    for i in range(len(action_batch)):
        gammablock = torch.zeros([agent.output_size, agent.output_size])
        a = action_batch[i]  # get the index of the action at that step
        gammablock[a, a] = gains_norm[i]
        gamma_known[i * agent.output_size:i * agent.output_size + agent.output_size,
                    i * agent.output_size:i * agent.output_size + agent.output_size] = gammablock

    pi_inv_known = torch.zeros([len(action_batch) * agent.output_size], dtype=torch.float64)
    for i in range(len(state_batch)):
        probs = agent(state_batch[i].unsqueeze(0))
        invprobs = 1 / (probs + 1E-8)
        pi_inv_known[i * agent.output_size:i * agent.output_size + agent.output_size] = invprobs

    pi_inv_safety_diag = torch.zeros([num_safe_states * agent.output_size, num_safe_states], dtype=torch.float64)
    for i in range(num_safe_states):
        probs = agent(safe_state_batch[i].unsqueeze(0))
        invprobs = 1 / (probs + 1E-8)
        a = safe_action_batch[i]  # get the index of the action at that step
        pi_inv_safety_diag[i * agent.output_size + a, i] = invprobs[0][a]

    # Inequalities: Ax >= B
    B = (dpi_des - torch.mv(torch.matmul(theta_safe_lower_left_block, gamma_known),
                            pi_inv_known) * learning_rate).unsqueeze(0).transpose(0, 1)
    A = torch.matmul(theta_safe_lower_right_block, pi_inv_safety_diag) * learning_rate

    P = np.eye(num_safe_states)
    Q = np.zeros(num_safe_states)  # J = sum G^2

    A_ineq = -A.detach().numpy()
    B_ineq = -B.detach().numpy().reshape(num_safe_states)

    safety_gains = solve_qp(P, Q, A_ineq, B_ineq)
    if safety_gains is not None:
        learn_gains = torch.cat([gains_norm, torch.from_numpy(safety_gains).squeeze()])
    else:
        learn_gains = gains_norm
    return learn_gains


# Compute the gains for the regional constraints
def compute_policy_evolution_safety_region(agent, state_batch, action_batch, gain_batch, safe_regions_states,
                                           safe_regions_actions, learning_rate, device):

    gains_norm = (gain_batch - gain_batch.mean()) / (gain_batch.std() + 1E-8)
    gamma_known = torch.zeros([len(action_batch) * agent.output_size, len(action_batch) * agent.output_size],
                              dtype=torch.float64)
    for i in range(len(action_batch)):
        gammablock = torch.zeros([agent.output_size, agent.output_size])
        a = action_batch[i]  # get the index of the action at that step
        gammablock[a, a] = gains_norm[i]
        gamma_known[i * agent.output_size:i * agent.output_size + agent.output_size,
                    i * agent.output_size:i * agent.output_size + agent.output_size] = gammablock

    pi_inv_known = torch.zeros([len(action_batch)*agent.output_size], dtype=torch.float64)
    for i in range(len(state_batch)):
        probs = agent(state_batch[i].unsqueeze(0))
        invprobs = 1 / (probs + 1E-8)
        pi_inv_known[i*agent.output_size:i*agent.output_size+agent.output_size] = invprobs

    num_safe_regions = len(safe_regions_states)
    num_states = len(state_batch[0])
    xidx = np.zeros(num_safe_regions, dtype=np.int)
    safe_state_batch = torch.zeros([num_safe_regions, num_states])
    safe_action_batch = np.zeros(num_safe_regions, dtype=np.int)
    gsx = []

    # For every safe region
    for i in range(num_safe_regions):
        # Check where the largest policy deviation is
        num_safe_states = len(safe_regions_states[i])
        g0 = 0
        for j in range(num_safe_states):
            dpi_des = compute_delta_pi_for_safety(agent, safe_regions_states[i][j].unsqueeze(0),
                                                  safe_regions_actions[i], 1)
            theta_safe = agent.compute_neural_tangent_kernel(
                torch.cat([state_batch, safe_regions_states[i][j].unsqueeze(0)])).to(device)
            theta_safe_lower_left_block_0 = theta_safe[state_batch.shape[0] * agent.output_size:theta_safe.shape[0],
                                                       0:state_batch.shape[0] * agent.output_size]
            theta_safe_lower_right_block_0 = theta_safe[state_batch.shape[0] * agent.output_size:theta_safe.shape[0],
                                                        state_batch.shape[0] * agent.output_size:theta_safe.shape[1]]

            theta_safe_lower_left_block = torch.zeros([1, theta_safe_lower_left_block_0.shape[1]], dtype=torch.float64)
            theta_safe_lower_right_block = torch.zeros([1, theta_safe_lower_right_block_0.shape[1]],
                                                       dtype=torch.float64)
            theta_safe_lower_left_block[0, :] = theta_safe_lower_left_block_0[safe_regions_actions[i][j], :]
            theta_safe_lower_right_block[0, :] = theta_safe_lower_right_block_0[safe_regions_actions[i][j], :]

            pi_inv_safety_diag = torch.zeros([1 * agent.output_size, 1], dtype=torch.float64)
            probs = agent(safe_regions_states[i][j].unsqueeze(0))
            invprobs = 1 / (probs + 1E-8)
            a = safe_regions_actions[i][j]  # get the index of the action at that step
            pi_inv_safety_diag[a, 0] = invprobs[0][a]

            # Linear equation system: Ax = B
            B = (dpi_des - torch.mv(torch.matmul(theta_safe_lower_left_block, gamma_known),
                                    pi_inv_known) * learning_rate).unsqueeze(0).transpose(0, 1)
            A = torch.matmul(theta_safe_lower_right_block, pi_inv_safety_diag) * learning_rate

            P = np.eye(1)
            Q = np.zeros(1)  # J = sum G^2

            A = -A.detach().numpy()
            B = -B.detach().numpy().reshape(1)

            safety_gain = solve_qp(P, Q, A, B)

            # log the G for the first region
            if i == 0:
                gsx.append(safety_gain)
                print(safety_gain)
            if safety_gain is not None and abs(safety_gain) > abs(g0):
                g0 = safety_gain
                xidx[i] = j

        # Append the constraints
        safe_state_batch[i] = safe_regions_states[i][xidx[i]]
        safe_action_batch[i] = safe_regions_actions[i][0]

    learn_gains = compute_policy_evolution_safety_ineq(agent, state_batch, action_batch, gain_batch, safe_state_batch,
                                                       safe_action_batch, learning_rate, device)
    return learn_gains, safe_state_batch, safe_action_batch, gsx


# Compute the gains for the regional constraints
def compute_policy_evolution_safety_region_simplified(agent, state_batch, action_batch, gain_batch, safe_regions_states,
                                                      safe_regions_actions, learning_rate, device):

    num_safe_regions = len(safe_regions_states)
    num_states = len(state_batch[0])
    xidx = np.zeros(num_safe_regions, dtype=np.int)
    safe_state_batch = torch.zeros([num_safe_regions, num_states])
    safe_action_batch = np.zeros(num_safe_regions, dtype=np.int)
    dpx = []

    # For every safe region
    for i in range(num_safe_regions):
        # Check where the largest policy deviation is
        num_safe_states = len(safe_regions_states[i])
        dpi_max = 0
        for j in range(num_safe_states):
            dpi_des = compute_delta_pi_for_safety(agent, safe_regions_states[i][j].unsqueeze(0),
                                                  safe_regions_actions[i], 1).item()
            if i == 0:
                dpx.append(dpi_des)
            if dpi_des > dpi_max:
                dpi_max = dpi_des
                xidx[i] = j

        # Append the constraints
        safe_state_batch[i] = safe_regions_states[i][xidx[i]]
        safe_action_batch[i] = safe_regions_actions[i][0]

    learn_gains = compute_policy_evolution_safety_ineq(agent, state_batch, action_batch, gain_batch, safe_state_batch,
                                                       safe_action_batch, learning_rate, device)
    return learn_gains, safe_state_batch, safe_action_batch, dpx


def safe_disk_cartpole(fi0, dfi0, r, a):

    safe_state_batch = []
    safe_action_batch = []

    for rho in range(5):
        for tau in range(6):
            fi = r/5*rho*np.cos(np.deg2rad(tau*60)).item() + fi0
            dfi = r/5*rho*np.sin(np.deg2rad(tau*60)).item() + dfi0

            safe_state_batch.append([fi, dfi])  # simplified 2-state version
            safe_action_batch.append(a)

    safe_state_batch = torch.tensor(safe_state_batch)
    safe_action_batch = np.array(safe_action_batch)
    return safe_state_batch, safe_action_batch


def safe_triangle_cartpole(a):

    safe_state_batch = []
    safe_action_batch = []
    if a == 0:
        m = -8
        b = -2
        for p in range(9):
            for y in range(11):
                for x in range(5):
                    pos = p/2 - 2
                    dfi = y/40 - 0.125
                    fi = (dfi - b)/m - x*0.01
                    safe_state_batch.append([pos, 0, fi, dfi])  # simplified 2-state version
                    safe_action_batch.append(a)
    if a == 1:
        m = -8
        b = 1.2
        for p in range(9):
            for y in range(11):
                for x in range(5):
                    pos = p / 2 - 2
                    dfi = y/40 - 0.125
                    fi = (dfi - b)/m + x*0.01
                    safe_state_batch.append([pos, 0, fi, dfi])  # simplified 2-state version
                    safe_action_batch.append(a)

    safe_state_batch_a = torch.tensor(safe_state_batch[0:100])
    safe_state_batch_b = torch.tensor(safe_state_batch[100:200])
    safe_state_batch_c = torch.tensor(safe_state_batch[200:300])
    safe_state_batch_d = torch.tensor(safe_state_batch[300:400])
    safe_state_batch_e = torch.tensor(safe_state_batch[400:-1])
    safe_action_batch_a = np.array(safe_action_batch[0:100])
    safe_action_batch_b = np.array(safe_action_batch[100:200])
    safe_action_batch_c = np.array(safe_action_batch[200:300])
    safe_action_batch_d = np.array(safe_action_batch[300:400])
    safe_action_batch_e = np.array(safe_action_batch[400:-1])
    return [safe_state_batch_a, safe_state_batch_b, safe_state_batch_c, safe_state_batch_d, safe_state_batch_e], \
           [safe_action_batch_a, safe_action_batch_b, safe_action_batch_c, safe_action_batch_d, safe_action_batch_e]


# Safety constraints (equalities/inequalities)
def safe_states_cartpole():

    safe_state_batch = torch.tensor(
        [[0, 0, 0.25, 0.05], [0, 0, -0.25, -0.05], [-0.5, 0, 0.25, 0.05], [0.5, 0, 0.25, 0.05], [0.5, 0, -0.25, -0.05],
         [-0.5, 0, -0.25, -0.05], [-1, 0, 0.25, 0.05], [1, 0, 0.25, 0.05], [1, 0, -0.25, -0.05], [-1, 0, -0.25, -0.05],
         [-1.5, 0, 0.25, 0.05], [1.5, 0, 0.25, 0.05], [1.5, 0, -0.25, -0.05], [-1.5, 0, -0.25, -0.05],
         [-2, 0, 0.25, 0.05], [2, 0, 0.25, 0.05], [2, 0, -0.25, -0.05], [-2, 0, -0.25, -0.05]])
    safe_action_batch = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    return safe_state_batch, safe_action_batch
