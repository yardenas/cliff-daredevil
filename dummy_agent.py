from collections import defaultdict

from gym.wrappers import TimeLimit

from cliff_daredevil.cliff_daredevil import CliffDarvedevil


def do_episode(agent, environment, render=True):
    observation, reset_info = environment.reset()
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    while not done:
        action = agent(observation)
        next_observation, reward, done, truncated, info = environment.step(action)
        terminal = done and not info.get("TimeLimit.truncated")
        observation = next_observation
        # if render:
        #     episode_summary["image"].append(environment.render(mode="rgb_array"))
        episode_summary["observation"].append(observation)
        episode_summary["next_observation"].append(next_observation)
        episode_summary["action"].append(action)
        episode_summary["reward"].append(reward)
        episode_summary["terminal"].append(terminal)
        episode_summary["info"].append(info)
    return steps, episode_summary


def main():
    env = CliffDarvedevil(render_mode='human')
    env = TimeLimit(env, max_episode_steps=100)
    do_episode(lambda *_: env.action_space.sample(), env)


if __name__ == "__main__":
    main()
