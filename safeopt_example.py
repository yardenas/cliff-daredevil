import argparse
from collections import defaultdict

import GPy
import matplotlib.pyplot as plt
import numpy as np
import safeopt
from gym.wrappers import TimeLimit

from cliff_daredevil import CliffDarvedevil


class Actor(object):
    def __init__(self, initial_policy_params, sum_cost_threshold):
        self._policy_params = initial_policy_params
        self._sum_cost_threshold = sum_cost_threshold
        self.objective_gp = None
        self.cost_gp = None
        self.optimizer = None

    def __call__(self, observation, optimal=False):
        # The agent here controls only the gas.
        params = self._policy_params if not optimal else [25.6, 0.0]
        hidden = -1.0 * observation[0] + params[0] + observation[1] * params[1]
        hidden = np.maximum(hidden, 0.0)
        gas = 10.0 * hidden
        return [np.tanh(gas), 0.0]

    def update_model(self, y):
        x = self._policy_params
        if not self.objective_gp and not self.cost_gp:
            # Bounds on the inputs variable
            bounds = [(0.0, 50.0), (-1.0, 1.0)]
            # Define Kernel
            kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2.0, lengthscale=[2.0, 0.1],
                                  ARD=True)
            kernel2 = GPy.kern.RBF(input_dim=len(bounds), variance=1.0, lengthscale=[2.0, 0.1],
                                   ARD=True)
            self.objective_gp = GPy.models.GPRegression(x[None], y[0, None, None], kernel)
            self.cost_gp = GPy.models.GPRegression(x[None], y[1, None, None], kernel2)
            parameter_set = safeopt.linearly_spaced_combinations(bounds, 100)
            self.optimizer = safeopt.SafeOpt([self.objective_gp, self.cost_gp], parameter_set,
                                             [-np.inf, 0.0],
                                             lipschitz=None,
                                             threshold=[0.1])
        else:
            self.optimizer.add_new_data_point(x, y)
        self._policy_params = self.optimizer.optimize()


def do_episode(agent, environment, render=True):
    observation = environment.reset()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation)
        next_observation, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated')
        observation = next_observation
        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
    return steps, episode_summary


def sample(agent, environment, render=True):
    steps, episode_summary = do_episode(agent, environment, render)
    objective = max(episode_summary['reward'])
    # Max: do not bypass the cliff.
    cost = max(list(map(lambda x: x['cost'], episode_summary['info'])))
    return np.hstack([objective, -cost])


def training_loop(agent, environment, render):
    for i in range(1000):
        render_now = render and (i % 50 == 0)
        y = sample(agent, environment, render_now)
        if render_now:
            # plot_3d
            agent.optimizer.plot(100)
            plt.show()
        if y[1] <= 0:
            print("Fell off the cliff!")
        if np.abs(y[0] - 1.0) < 0.1 and y[1] > 0:
            print("Solved!")
            y = sample(agent, environment, True)
        print("Objective:", y[0])
        print("Safety objective:", y[1])
        agent.update_model(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--episode_length', default=600)
    parser.add_argument('--sum_cost_threshold', default=0.1)
    args = parser.parse_args()
    env = CliffDarvedevil()
    env = TimeLimit(env, max_episode_steps=args.episode_length)
    x0 = np.zeros([2, ])
    agent = Actor(x0, args.sum_cost_threshold)
    y0 = sample(agent, env, args.render)
    agent.update_model(y0)
    training_loop(agent, env, args.render)


if __name__ == "__main__":
    main()
