# ORIGINAL CODE PROVIDED BY wisnunugroho21 AT
# https://github.com/wisnunugroho21/reinforcement_learning_phasic_policy_gradient
# Software is distributed under a GPL-3.0 License.

import gym
from src.utils.env_loader import make_gym_env

from src.ppg.runner import RunnerRay
from src.ppg.model import Learner

from src.ppg.logging import EpochInfo, init_logging
from itertools import count as infinite_range
from dataclasses import asdict
from pprint import pformat

from pathlib import Path
import time
import datetime

from src.utils.args import get_args
import logging

import ray
import wandb

def main(args):

    wandb_run, continue_run, start_epoch, output_path = init_logging(args)

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
    ray.init()

    try:
        runners = [
            RunnerRay.remote(
                args.env,
                args.train_mode,
                args.visualize,
                args.n_update,
                i,
                save_path=output_path,
            )
            for i in range(args.num_workers)
        ]

        if not continue_run:
            learner.save_weights(output_path)

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, {}, 0))
            time.sleep(3)

        for epoch in infinite_range(start_epoch):

            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, reward_partials, eps_time, ep_info, tag = ray.get(ready)[0]

            episode_ids = not_ready
            episode_ids.append(
                runners[tag].run_episode.remote(
                    i_episode, total_reward, reward_partials, eps_time
                )
            )

            states, actions, action_means, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, action_means, rewards, dones, next_states)

            learner.update_ppo()
            if epoch % args.n_aux_update == 0:
                learner.update_aux()

            learner.save_weights(output_path)


            if epoch % 5 == 0:
                info = EpochInfo(
                    epoch,
                    time.time() - start,
                    ep_info
                )
                if args.log_wandb:
                    wandb.log(asdict(info), step=epoch)

                logging.info("Epoch information: {}".format(pformat(asdict(info))))

                ep_info = None
                info = None

    except KeyboardInterrupt:
        logging.warning("Training has been stopped.")
    finally:
        ray.shutdown()
        with open(Path(output_path) / "epoch.info", "w") as ep_file:
            ep_file.write(str(epoch))

        finish = time.time()
        timedelta = finish - start
        logging.info("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))

if __name__ == "__main__":
    args = get_args()

    main(args)
