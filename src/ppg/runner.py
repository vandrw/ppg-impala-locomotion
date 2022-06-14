import gym
from src.utils.env_loader import make_gym_env
from src.ppg.agent import Agent
import numpy as np


class Runner:
    def __init__(
        self,
        experiment_type,
        data_subject,
        initial_logstd,
        training_mode,
        render,
        n_update,
        tag,
        save_path,
    ):
        env_name = make_gym_env(
            experiment_type, data_subject=data_subject, visualize=render
        )
        self.env = gym.make(env_name, disable_env_checker=True)
        self.states = self.env.reset()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = Agent(
            self.state_dim, self.action_dim, initial_logstd, training_mode
        )

        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update
        self.max_action = 1.0

        self.save_path = save_path
        print("[Proc {}] Worker initialized.".format(tag))

    def run_episode(self, i_episode, total_reward, eps_time):
        self.agent.memory.clear_memory()
        self.agent.load_weights(self.save_path)
        ep_info = None

        for _ in range(self.n_update):
            action, action_mean, action_std = self.agent.act(self.states)

            action_gym = np.clip(action, 0.0, 1.0) * self.max_action
            next_state, reward, done, _ = self.env.step(action_gym)

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

                ep_info = {"total_reward": total_reward, "episode_time": eps_time}

                total_reward = 0
                eps_time = 0

        self.agent.memory.update_std(action_std)

        return (
            self.agent.get_all(),
            i_episode,
            total_reward,
            eps_time,
            ep_info,
        )
