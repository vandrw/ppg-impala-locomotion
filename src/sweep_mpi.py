from mpi4py import MPI

comm = MPI.COMM_WORLD
w_size = comm.Get_size()
rank = comm.Get_rank()

import gym
import time
from math import ceil
import traceback

from src.env_loader import make_gym_env
from src.args import get_args

SWEEP_TIME = 150

def main_worker(config):
    from src.ppg.agent import Agent
    import numpy as np
    class Runner:
        def __init__(
            self, experiment_type, training_mode, render, n_update, tag, save_path
        ):
            env_name = make_gym_env(experiment_type, visualize=render)
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
            print("[Proc {}] Worker initialized.".format(tag))

        def run_episode(self, i_episode, total_reward, eps_time):

            self.agent.load_weights(self.save_path)

            for _ in range(self.n_update):
                action, action_mean = self.agent.act(self.states)

                action_gym = np.clip(action, -1.0, 1.0) * self.max_action
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

                    info = {
                        "total_reward": total_reward, 
                        "episode_time": eps_time
                        }

                    total_reward = 0
                    eps_time = 0

            return (
                self.agent.get_all(),
                i_episode,
                total_reward,
                eps_time,
                info,
            )

    msg = None

    output_path = comm.bcast(msg, root=0)

    runner = Runner(
        config.env,
        config.train_mode,
        config.visualize,
        config.n_update,
        rank,
        save_path=output_path,
    )

    trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(rank, 0, 0)
    data = (trajectory, done_info)
    comm.send(data, dest=0)
    time.sleep(3)

    try:
        for _ in range(ceil(SWEEP_TIME / (w_size - 1))):
            trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(
                i_episode, total_reward, eps_time
            )

            data = (trajectory, done_info)

            comm.send(data, dest=0)
    except Exception as ex:
        print("Proc {} terminated".format(rank))
        traceback.print_exception(type(ex), ex, ex.__traceback__)


def main_head(config):
    from src.ppg.model import Learner

    from src.ppg.logging import EpochInfo
    from dataclasses import asdict
    from pathlib import Path
    import datetime

    import wandb

    output_path = Path("output") / config.run_name

    wandb_run = wandb.init(
        project="rug-locomotion-ppg", 
        config=config, 
        settings=wandb.Settings(start_method="fork"),
        mode="offline",
        group="sweep"
        )

    env_name = make_gym_env(config.env, visualize=config.visualize)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = Learner(
        state_dim,
        action_dim,
        config.train_mode,
        config.policy_kl_range,
        config.policy_params,
        config.value_clip,
        config.entropy_coef,
        config.vf_loss_coef,
        config.batch_size,
        config.PPO_epochs,
        config.gamma,
        config.lam,
        config.learning_rate,
    )

    start = time.time()

    if (Path(output_path) / "agent.pth").exists():
        print("Another agent has initiated training with this configuration. Stopping...")
        comm.Abort()
        exit()

    learner.save_weights(output_path)

    msg = output_path

    output_path = comm.bcast(msg, root=0)

    try:
        for epoch in range(SWEEP_TIME + 1):
            data = None

            data = comm.recv()
            trajectory, done_info = data

            states, actions, action_means, rewards, dones, next_states = trajectory
            learner.save_all(
                states, actions, action_means, rewards, dones, next_states
            )

            learner.update_ppo()
            if epoch % config.n_aux_update == 0:
                learner.update_aux()

            learner.save_weights(output_path)

            if epoch % 5 == 0:
                info = EpochInfo(epoch, time.time() - start, done_info)
                if config.log_wandb:
                    wandb.log(asdict(info), step=epoch)
                
                print("Epoch {}/{}: {}".format(epoch, SWEEP_TIME, done_info))

                done_info = None
                info = None

    except Exception as ex:
        print("Main terminated")
        traceback.print_exception(type(ex), ex, ex.__traceback__)
    finally:
        finish = time.time()
        timedelta = finish - start
        print("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))
        wandb_run.finish()
        comm.Abort()

if __name__ == "__main__":
    config = get_args()

    if rank == 0:
        print("Number of agents available: ", w_size - 1)
        main_head(config)
    else:
        main_worker(config)
        
