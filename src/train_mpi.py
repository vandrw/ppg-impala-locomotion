# ORIGINAL CODE PROVIDED BY wisnunugroho21 AT
# https://github.com/wisnunugroho21/reinforcement_learning_phasic_policy_gradient
# Software is distributed under a GPL-3.0 License.

from mpi4py import MPI

comm = MPI.COMM_WORLD
w_size = comm.Get_size() - 1
rank = comm.Get_rank()

import gym
from src.utils.env_loader import make_gym_env

from itertools import count as infinite_range
import time
import traceback
import signal

from src.utils.args import get_args

interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True

# Register the signal handler
signal.signal(signal.SIGUSR1, signal_handler)


def main_worker(args):
    from src.ppg.runner import Runner

    msg = None
    output_path = comm.bcast(msg, root=0)

    runner = Runner(
        args.env,
        args.data,
        args.initial_logstd,
        args.train_mode,
        args.normalize_obs,
        args.obs_clip_range,
        args.visualize,
        args.n_steps,
        rank,
        save_path=output_path,
    )

    trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(rank, 0, 0)

    if args.normalize_obs:
        data = (trajectory, done_info, runner.normalizer.mean, runner.normalizer.var)
    else:
        data = (trajectory, done_info)
    comm.send(data, dest=0)

    try:
        for _ in infinite_range(0):
            trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(
                i_episode, total_reward, eps_time
            )

            if args.normalize_obs:
                data = (trajectory, done_info, runner.normalizer.mean, runner.normalizer.var)
            else:
                data = (trajectory, done_info)
            comm.send(data, dest=0)
            
            if interrupted:
                break
    except Exception as ex:
        print("Proc {} terminated".format(rank))
        traceback.print_exception(type(ex), ex, ex.__traceback__)


def main_head(args):
    from src.ppg.learner import Learner

    from src.utils.logging import EpochInfo, init_logging
    from dataclasses import asdict

    from pathlib import Path
    import datetime
    import logging

    import wandb

    wandb_run, continue_run, start_epoch, output_path = init_logging(args)

    env_name = make_gym_env(args.env, args.data, visualize=args.visualize)

    env = gym.make(env_name, disable_env_checker=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = Learner(
        state_dim,
        action_dim,
        args.train_mode,
        args.normalize_obs,
        args.obs_clip_range,
        args.ppo_kl_range,
        args.slope_rollback,
        args.slope_likelihood,
        args.val_clip_range,
        args.entropy_coef,
        args.vf_loss_coef,
        args.ppo_batchsize,
        args.ppo_epochs,
        args.aux_batchsize,
        args.aux_epochs,
        args.beta_clone,
        args.gamma,
        args.lambd,
        args.learning_rate,
        args.initial_logstd
    )

    del env

    start = time.time()
    if not continue_run:
        learner.save_weights(output_path)
        if args.normalize_obs:
            learner.normalizer.save(output_path)
    else:
        learner.load_weights(output_path)
        logging.info("Loaded previous Learner weights!")
        if args.normalize_obs:
            try:
                learner.normalizer.load(output_path)
                logging.info("Loaded previous Normalizer!")
            except:    
                logging.error("The previous run did not use a normalizer.")
                logging.warning("Creating a new normalizer. Be aware that the " 
                                + "policy is not trained for non-normalized states.")
                learner.normalizer.save(output_path)

    msg = output_path
    output_path = comm.bcast(msg, root=0)

    logging.info("Starting training... Workers available: {}".format(w_size))
    try:
        avg_reward = 0
        avg_ep_time = 0
        data = None

        # Use the first set of results from each worker to update the normalizer.
        # Do not use these results for training, since it will update the policy
        # with the normal states.
        if args.normalize_obs and not continue_run:
            logging.info("Initializing normalizer. Using the first set of states from each worker...")
            for t in range(w_size):
                data = comm.recv()
                _, done_info, w_norm_mean, w_norm_var = data

                learner.normalizer.update(w_norm_mean, w_norm_var, args.n_steps)
                avg_reward += done_info["total_reward"]
                avg_ep_time += done_info["episode_time"]
            
            logging.info(
                    "Normalizer update phase: {} trajectories, reward {}, episode time {}".format(
                        t + 1, avg_reward / w_size, avg_ep_time / w_size
                    )
                )
            learner.normalizer.save(output_path)
            avg_reward = 0
            avg_ep_time = 0

        for epoch in infinite_range(start_epoch):

            data = comm.recv()
            
            if args.normalize_obs:
                trajectory, done_info, w_norm_mean, w_norm_var = data
                learner.normalizer.update(w_norm_mean, w_norm_var, args.n_steps)
            else:
                trajectory, done_info = data

            states, actions, action_means, action_std, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, action_means, action_std, rewards, dones, next_states)

            learner.update_ppo()
            if epoch % args.aux_update == 0:
                learner.update_aux()

            learner.save_weights(output_path)
            if args.normalize_obs:
                learner.normalizer.save(output_path)

            avg_reward += done_info["total_reward"]
            avg_ep_time += done_info["episode_time"]

            if (epoch + 1) % w_size == 0:
                real_epoch = int(epoch / w_size)
                avg_reward /= w_size
                avg_ep_time /= w_size

                if args.log_wandb:
                    wandb.log(
                        asdict(
                            EpochInfo(
                                epoch + 1,
                                time.time() - start,
                                avg_reward,
                                avg_ep_time
                            )
                        ),
                        step=real_epoch,
                    )

                logging.info(
                    "Epoch {} (trajectory {}): reward {}, episode time {}".format(
                        real_epoch, epoch + 1, avg_reward, avg_ep_time
                    )
                )

                avg_reward = 0
                avg_ep_time = 0

            if interrupted:
                logging.warning("Termination signal received. Finishing...")
                break

    except Exception:
        print("Main terminated")
        print(traceback.format_exc())
    finally:
        logging.warning("Training has been stopped.")
        if wandb_run:
            wandb_run.finish()

        with open(Path(output_path) / "epoch.info", "w") as ep_file:
            ep_file.write(str(epoch))

        finish = time.time()
        timedelta = finish - start
        logging.info("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))
        
        comm.Abort()


if __name__ == "__main__":
    args = get_args()

    args.num_workers = w_size

    if rank == 0:
        main_head(args)
    else:
        main_worker(args)
