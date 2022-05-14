from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import gym
from src.env_loader import make_gym_env
import time

from src.args import get_args

SWEEP_TIME = 250

def main_worker(args):
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
        args.env,
        args.train_mode,
        args.visualize,
        args.n_update,
        rank,
        save_path=output_path,
    )

    trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(rank, 0, 0)
    data = (trajectory, done_info)
    comm.send(data, dest=0)
    time.sleep(3)

    try:
        for _ in range(SWEEP_TIME):
            trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(
                i_episode, total_reward, eps_time
            )

            data = (trajectory, done_info)

            comm.send(data, dest=0)
    except KeyboardInterrupt:
        pass


def main_head(args):
    from src.ppg.model import Learner

    from src.ppg.logging import EpochInfo, init_output
    from dataclasses import asdict
    from pprint import pformat

    from pathlib import Path
    import datetime
    import logging

    import wandb

    wandb.init(project="rug-locomotion-ppg", settings=wandb.Settings(start_method="fork"))
    output_path = init_output(args)

    env_name = make_gym_env(args.env, visualize=args.visualize)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = Learner(
        state_dim,
        action_dim,
        args.train_mode,
        args.policy_kl_range,
        args.policy_params,
        args.value_clip,
        args.entropy_coef,
        args.vf_loss_coef,
        args.batch_size,
        args.PPO_epochs,
        args.gamma,
        args.lam,
        args.learning_rate,
    )

    start = time.time()
    learner.save_weights(output_path)

    msg = output_path

    output_path = comm.bcast(msg, root=0)

    try:
        for epoch in range(SWEEP_TIME):
            data = None

            data = comm.recv()
            trajectory, done_info = data

            states, actions, action_means, rewards, dones, next_states = trajectory
            learner.save_all(
                states, actions, action_means, rewards, dones, next_states
            )

            learner.update_ppo()
            if epoch % args.n_aux_update == 0:
                learner.update_aux()

            learner.save_weights(output_path)

            if epoch % 5 == 0:
                info = EpochInfo(epoch, time.time() - start, done_info)
                if args.log_wandb:
                    wandb.log(asdict(info), step=epoch)

                logging.info("Epoch information: {}".format(pformat(asdict(info))))

                done_info = None
                info = None

    except KeyboardInterrupt:
        logging.warning("Training has been stopped.")
    finally:
        with open(Path(output_path) / "epoch.info", "w") as ep_file:
            ep_file.write(str(epoch))

        finish = time.time()
        timedelta = finish - start
        logging.info("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))

        MPI.Finalize()


if __name__ == "__main__":
    args = get_args()
    
    if rank == 0:
        main_head(args)
    else:
        main_worker(args)
