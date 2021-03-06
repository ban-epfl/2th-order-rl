import numpy as np

batch_size = 1
import gym

env = gym.make('CartPole-v0')


class LogisticPolicy:

    def __init__(self, θ, α, γ):
        # Initialize paramters θ, learning rate α and discount factor γ

        self.θ = θ
        self.α = α
        self.γ = γ

    def logistic(self, y):
        # definition of logistic function

        return 1 / (1 + np.exp(-y))

    def probs(self, x):
        # returns probabilities of two actions

        y = x @ self.θ
        prob0 = self.logistic(y)

        return np.array([prob0, 1 - prob0])

    def act(self, x):
        # sample an action in proportion to probabilities

        probs = self.probs(x)
        action = np.random.choice([0, 1], p=probs)

        return action, probs[action]

    def grad_log_p(self, x):
        # calculate grad-log-probs

        y = x @ self.θ
        grad_log_p0 = x - x * self.logistic(y)
        grad_log_p1 = - x * self.logistic(y)

        return grad_log_p0, grad_log_p1

    def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
        # dot grads with future rewards for each action in episode

        return grad_log_p.T @ discounted_rewards

    def discount_rewards(self, rewards):
        # calculate temporally adjusted, discounted rewards

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.γ + rewards[i]
            discounted_rewards[i] = cumulative_rewards

        return discounted_rewards

    def update(self, rewards, obs, actions):
        # calculate gradients for each action over all observations
        grad_log_p = np.array([self.grad_log_p(ob)[action] for ob, action in zip(obs, actions)])

        assert grad_log_p.shape == (len(obs), 4)

        # calculate temporaly adjusted, discounted rewards
        discounted_rewards = self.discount_rewards(rewards)

        # gradients times rewards
        dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)
        return dot


def run_episode(env, policy, render=False):
    observation = env.reset()
    totalreward = 0

    observations = []
    actions = []
    rewards = []
    probs = []

    done = False

    while not done:
        if render:
            env.render()

        observations.append(observation)

        action, prob = policy.act(observation)
        observation, reward, done, info = env.step(action)

        totalreward += reward
        rewards.append(reward)
        actions.append(action)
        probs.append(prob)

    return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)


def train(θ, α, γ, Policy, MAX_EPISODES=1000, seed=None, evaluate=False):
    # initialize environment and policy
    env = gym.make('CartPole-v0')
    if seed is not None:
        env.seed(seed)
    episode_rewards = []
    iter_grads = []
    iter_reward = 0
    policy = Policy(θ, α, γ)
    gradient_of_the_episode = 0
    # train until MAX_EPISODES
    for i in range(MAX_EPISODES):

        # run a single episode
        total_reward, rewards, observations, actions, probs = run_episode(env, policy)
        iter_reward += total_reward

        # update policy
        gradient_of_the_episode += policy.update(rewards, observations, actions)
        if (i + 1) % batch_size == 0:
            # keep track of episode rewards
            episode_rewards.append(iter_reward / batch_size)
            # gradient ascent on parameters
            iter_grads.append(gradient_of_the_episode / batch_size)
            policy.θ += policy.α * iter_grads[-1]
            gradient_of_the_episode = 0
            iter_reward=0
            print("iteration: " + str((i + 1) // batch_size) + " Score: " + str(total_reward) + " ")

    # # evaluation call after training is finished - evaluate last trained policy on 100 episodes
    # if evaluate:
    #     env = Monitor(env, 'pg_cartpole/', video_callable=False, force=True)
    #     for _ in range(100):
    #         run_episode(env, policy, render=False)
    #     env.env.close()

    return iter_grads, episode_rewards, policy


# additional imports for saving and loading a trained policy
from gym.wrappers.monitor import Monitor, load_results

# for reproducibility
GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)

iter_grads, episode_rewards, policy = train(θ=np.random.rand(4),
                                            α=0.001,
                                            γ=0.99,
                                            Policy=LogisticPolicy,
                                            MAX_EPISODES=6000*batch_size,
                                            seed=GLOBAL_SEED,
                                            evaluate=False)

import matplotlib.pyplot as plt

plt.plot(episode_rewards)

print("means of rewards: ", np.mean(episode_rewards))
print("std of rewards: ", np.std(episode_rewards))
print("std of grads: ", np.std(iter_grads))

plt.show()
