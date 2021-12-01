from PolicyNet import NeuralNet
from Functions import select_action, compute_gain, compute_logprobs, compute_policy_evolution_safety_region_simplified,\
    compute_policy_evolution_safety_eq, compute_policy_evolution_safety_ineq, safe_states_cartpole, \
    safe_triangle_cartpole, compute_learning_stats, plot_data
import gym
import torch
import numpy as np
import pickle

# setting device on GPU if available, else CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU available.")
else:
    device = torch.device('cpu')
    print("GPU not available.")

# 0: equality
# 1: inequality
# 2: regional
CONSTRAINT_TYPE = 1


def main():

    # Set up the environment
    env = gym.make('CartPole-v0')
    state_num = 4
    action_num = 2
    num_episodes = 200
    num_seeds = 5
    num_constr_ep = 50
    discount_factor = 1
    n_win_ticks = 195

    learning_rate = 0.0001
    hidden_width = 5000

    # Set up the constraints
    if CONSTRAINT_TYPE == 0:  # Equality
        safe_state_batch, safe_action_batch = safe_states_cartpole()
    elif CONSTRAINT_TYPE == 1:  # Inequality
        safe_state_batch, safe_action_batch = safe_states_cartpole()
    elif CONSTRAINT_TYPE == 2:  # Regional
        safe_state_batch0, safe_action_batch0 = safe_triangle_cartpole(0)
        safe_state_batch1, safe_action_batch1 = safe_triangle_cartpole(1)
        safe_state_batch = safe_state_batch0 + safe_state_batch1
        safe_action_batch = safe_action_batch0 + safe_action_batch1
    else:
        safe_state_batch = []
        safe_action_batch = []

    # Logging
    gains_logged = []
    avg_gains_logged = []
    loss_logged = []

    # Random seeds
    for i in range(num_seeds):

        # Create the agent
        agent = NeuralNet(state_num, action_num, hidden_width, learning_rate)
        episode_gains = []
        mean_score = []
        e = 0

        # Run a trajectory with the current policy
        while e < num_episodes:
            reward_batch = []
            logprob_batch = []
            action_batch = []
            state_batch = []

            state = env.reset()
            done = False

            while not done:
                action, log_prob = select_action(agent, state)
                state_batch.append(torch.from_numpy(state).float().unsqueeze(0))

                state, reward, done, _ = env.step(action)
                reward_batch.append(reward)
                action_batch.append(action)
                logprob_batch.append(log_prob)

            # Update the policy
            gain_batch = compute_gain(reward_batch, discount_factor)
            state_batch_tensor = torch.cat(state_batch)

            if e < num_constr_ep:  # to speed up learning, train only for the first x episodes.
                if CONSTRAINT_TYPE == 0:  # Equality
                    gain_learn = compute_policy_evolution_safety_eq(agent, state_batch_tensor, action_batch,
                                                                    gain_batch, safe_state_batch, safe_action_batch,
                                                                    device)
                    safe_logprobs = compute_logprobs(agent, safe_state_batch, safe_action_batch)
                elif CONSTRAINT_TYPE == 1:  # Inequality
                    gain_learn = compute_policy_evolution_safety_ineq(agent, state_batch_tensor, action_batch,
                                                                      gain_batch, safe_state_batch, safe_action_batch,
                                                                      learning_rate, device)
                    safe_logprobs = compute_logprobs(agent, safe_state_batch, safe_action_batch)
                elif CONSTRAINT_TYPE == 2:  # Regional
                    gain_learn, safe_state_batch_reg, safe_action_batch_reg, dpx_ = \
                        compute_policy_evolution_safety_region_simplified(agent, state_batch_tensor, action_batch,
                                                                          gain_batch, safe_state_batch,
                                                                          safe_action_batch, learning_rate, device)
                    safe_logprobs = compute_logprobs(agent, safe_state_batch_reg, safe_action_batch_reg)
                else:
                    gain_learn = []
                    safe_state_batch = []
                    safe_action_batch = []
                    safe_logprobs = []

                logprob_learn = logprob_batch + safe_logprobs

                # Train with the safety appended data
                agent.train_network(logprob_learn, gain_learn, 1)

            episode_gains.append(len(reward_batch))
            print("Episode ", e, " gain: ", len(reward_batch))

            # Check if won
            if e > 100:
                mean_score.append(np.mean(episode_gains[-100:]))
                if mean_score[-1] >= n_win_ticks:
                    print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
            else:
                mean_score.append(np.mean(episode_gains))
            e = e + 1
        loss_logged.append(agent.loss_arr)
        gains_logged.append(episode_gains)
        avg_gains_logged.append(mean_score)

        # Save the trained agent for the last seed.
        if e == num_episodes and i == num_seeds-1:
            pickle.dump(agent, open("cartpole_agent_ineq.p", "wb"))

        del agent

    env.close()

    # Plot learning stats
    data = compute_learning_stats(avg_gains_logged)
    plot_data(data, n_win_ticks)


if __name__ == '__main__':
    main()
