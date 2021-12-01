import pickle
import gym
import torch
import matplotlib.pyplot as plt


def plot_data(data, average, win):
    ax = plt.axes()
    x = torch.linspace(0, len(data), len(data))
    win = [win] * len(data)
    ax.plot(x, data, label="Episode score")
    ax.plot(x, win, label="Pass threshold")
    ax.plot(x, average, label="Average score")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Score')
    plt.legend()
    plt.show()


def select_action(agent, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = agent(state)  # Output of the NN (probabilities due to softmax nonlinearity
    m = torch.distributions.Categorical(probs)  # Create a distribution with probabilities from the agent's output
    action = m.sample()  # Sample one (index) from the policy
    log_prob = m.log_prob(action)
    return action.item(), log_prob


def run_episode(env, agent):

    state = env.reset()
    done = False
    episode_reward = 0

    while not done:  # Don't infinite loop while learning
        action, log_prob = select_action(agent, state)
        state, reward, done, _ = env.step(action)
        episode_reward = episode_reward + reward

    return episode_reward


def main():

    env = gym.make('CartPole-v0')
    agent = pickle.load(open("cartpole_agent_ineq.p", "rb"))

    win_condition = 195
    num_episodes = 200
    episode_scores = []
    mean_score = []

    for e in range(num_episodes):
        episode_scores.append(run_episode(env, agent))
        print("Episode ", e, " score: ", episode_scores[e])
        if e > 100:
            mean_score.append(torch.mean(torch.tensor(episode_scores[-100:])))
        else:
            mean_score.append(torch.mean(torch.tensor(episode_scores)))

    print("Average score:", mean_score[-1])
    plot_data(episode_scores, mean_score, win_condition)


if __name__ == '__main__':
    main()
