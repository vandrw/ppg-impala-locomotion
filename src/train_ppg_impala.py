import gym
from src.env_loader import load_ppg_env

from src.ppg.runner import Runner
from src.ppg.model import Learner
from src.ppg.logging import EpochInfo, DistributionInfo

from dataclasses import asdict
from pprint import pformat

import time
import datetime
import os
import sys

import argparse
import yaml
import logging

import ray
import wandb


def init_output(run_name):
    output_path = os.path.join("output", run_name)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    return output_path


def main(args):
    output_path = init_output(args.run_name)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s : %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_path, "train.log"), mode="w"),
            stream_handler,
        ],
    )

    logging.info("Saving configuration in {}/{}.".format(output_path, "train.log"))
    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        f.write(yaml.safe_dump(args.__dict__, default_flow_style=False))

    continue_run = os.path.exists((os.path.join(output_path, "agent.pth")))
    if continue_run:
        logging.info(
            "Found previous model in {}. Continuing training.".format(output_path)
        )

    if args.log_wandb:
        try:
            wandb.init(
                project="rug-locomotion-ppg",
                config=args,
                name=args.run_name,
                id=args.run_name,
                resume="must" if continue_run else "never",
            )
        except ModuleNotFoundError:
            logging.error(
                "You've requested to log metrics to wandb but package was not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    env_name = load_ppg_env(args.env, visualize=args.visualize)

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

    t_aux_updates = 0
    start = time.time()
    ray.init()

    try:
        runners = [
            Runner.remote(
                args.env,
                args.train_mode,
                args.visualize,
                args.n_update,
                i,
                save_path=output_path,
            )
            for i in range(args.n_agent)
        ]

        if not continue_run:
            learner.save_weights(output_path)

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(3)

        for _ in range(1, args.n_episode + 1):

            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            episode_ids = not_ready
            episode_ids.append(
                runners[tag].run_episode.remote(
                    i_episode,
                    total_reward,
                    eps_time
                )
            )

            states, actions, action_means, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, action_means, rewards, dones, next_states)

            learner.update_ppo()
            if t_aux_updates == args.n_aux_update:
                learner.update_aux()
                t_aux_updates = 0

            learner.save_weights(output_path)

            # Logging Epoch results
            # info_returns = DistributionInfo.from_array(ep_returns)
            # info_lens = DistributionInfo.from_array(ep_lens)
            # info_dist = DistributionInfo.from_array(epoch_ep_distances)
            # info_reward_partials = {
            #     key: DistributionInfo.from_array(reward_partials[key])
            #     for key in reward_partials
            # }
            # info = EpochInfo(
            #     epoch,
            #     time() - start,
            #     info_returns,
            #     info_lens,
            #     info_dist,
            #     info_reward_partials,
            # )
            # if args.log_wandb:
            #     wandb.log(asdict(info))
            
            # logging.info("Epoch information: {}".format(pformat(asdict(info))))

    except KeyboardInterrupt:
        logging.warning("Training has been stopped.")
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        logging.info("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))


def _parse_args(parser, config_parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


if __name__ == "__main__":
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(
        description="Training Configuration", add_help=False
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        default="",
        help="Path to the YAML config file specifying the default parameters.",
    )
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of Phasic Policy Gradient for training Physics based Musculoskeletal Models."
    )
    parser.add_argument(
        "--env",
        help="The environment type used. Currently supports the healthy and prosthesis models.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Display name to use in wandb. Also used for the path to save the model. Provide a unique name.",
    )
    parser.add_argument(
        "--log-wandb", action="store_true", help="Whether to save output on wandb."
    )
    parser.add_argument(
        "--train-mode",
        action="store_true",
        help="Whether to save new checkpoints during learning. If used, there will be changes made to the model.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the environment. Disregard this argument if you run this on Peregrine/Google Collab.",
    )
    parser.add_argument(
        "--n-update",
        type=int,
        default=1024,
        help="How many episodes before the Policy is updated. Also regarded as the simulation budget (steps) per iteration.",
    )
    parser.add_argument(
        "--n-aux-update",
        type=int,
        default=5,
        help="How many episodes before the Auxiliary is updated.",
    )
    parser.add_argument(
        "--n-episode",
        type=int,
        default=1000000,
        help="How many episodes you want to run.",
    )
    parser.add_argument(
        "--n-agent",
        type=int,
        default=4,
        help="How many agents you want to run asynchronously.",
    )
    parser.add_argument("--policy-kl-range", type=float, default=0.03, help="TODO.")
    parser.add_argument("--policy-params", type=int, default=5, help="TODO.")
    parser.add_argument("--value-clip", type=float, default=1.0, help="TODO.")
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="How much action randomness is introduced.",
    )
    parser.add_argument("--vf-loss-coef", type=float, default=1.0, help="TODO.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="How many batches per update. The number of batches is given by the number of updates divided by the batch size.",
    )
    parser.add_argument(
        "--PPO-epochs", type=int, default=10, help="How many PPO epochs per update."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="TODO.")
    parser.add_argument("--lam", type=float, default=0.95, help="TODO.")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="TODO.")

    args = _parse_args(parser, config_parser)

    assert (
        args.env is not None
    ), "Provide an environment type: 'healthy', 'prosthesis', 'terrain' "
    assert (
        args.run_name is not None
    ), "Provide an unique run name. If you wish to continue training, use the same name found in outputs/name_of_your_run."

    main(args)
