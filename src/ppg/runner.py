import gym
from src.env_loader import load_ppg_env

from src.ppg.agent import Agent

import numpy as np
import ray

@ray.remote
class Runner:
    def __init__(self, experiment_type, training_mode, render, n_update, tag, save_path):
        env_name = load_ppg_env(experiment_type, visualize=render)
        self.env = gym.make(env_name)
        self.states = self.env.reset()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = Agent(self.state_dim, self.action_dim, training_mode)

        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update
        self.max_action = 1.0

        self.save_path = save_path

    def run_episode(self, i_episode, total_reward, eps_time):

        self.agent.load_weights(self.save_path)
        # reward_partials = {}
        # ep_distances = []

        for _ in range(self.n_update):
            action, action_mean = self.agent.act(self.states)

            action_gym = np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, done, info = self.env.step(action_gym)

            eps_time += 1
            total_reward += reward

            if self.training_mode:
                self.agent.save_eps(
                    self.states.tolist(),
                    action,
                    action_mean,
                    reward,
                    float(done),
                    next_state.tolist(),
                )

            self.states = next_state

            if done:
                self.states = self.env.reset()
                i_episode += 1

                # epoch_ep_returns.append(total_reward)
                # epoch_ep_lens.append(eps_time)
                # epoch_ep_distances.append(info["dist_from_origin"])

                # for key in info:
                #     if key.startswith("reward"):
                #         if key not in reward_partials:
                #             reward_partials[key] = []
                #         reward_partials[key].append(info[key])

                # ep_distances.append(info["dist_from_origin"])
                # print(
                #     "Episode {} \t t_reward: {} \t time: {} \t process no: {}".format(
                #         i_episode, total_reward, eps_time, self.tag
                #     )
                # )

                total_reward = 0
                eps_time = 0
                # reward_partials = {}
                # ep_distances = []

        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag