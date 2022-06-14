from mpi4py import MPI

comm = MPI.COMM_WORLD
w_size = comm.Get_size() - 1
rank = comm.Get_rank()

import gym
from src.utils.env_loader import make_gym_env

from itertools import count as infinite_range
import time
import traceback

from src.utils.args import get_args

SWEEP_TIME = 50

def main_worker(args):
    from src.ppg.runner import Runner

    msg = None
    output_path = comm.bcast(msg, root=0)

    runner = Runner(
        args.env,
        args.data,
        args.initial_logstd,
        args.train_mode,
        args.visualize,
        args.n_steps,
        rank,
        save_path=output_path,
    )

    trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(rank, 0, 0)

    data = (trajectory, done_info)
    comm.send(data, dest=0)

    try:
        for _ in infinite_range(0):
            trajectory, i_episode, total_reward, eps_time, done_info = runner.run_episode(
                i_episode, total_reward, eps_time
            )

            data = (trajectory, done_info)
            comm.send(data, dest=0)
    except Exception as ex:
        print("Proc {} terminated".format(rank))
        traceback.print_exception(type(ex), ex, ex.__traceback__)


def main_head(args):
    from src.ppg.learner import Learner

    from src.utils.logging import EpochInfo
    from dataclasses import asdict

    from pathlib import Path
    import datetime

    import wandb

    output_path = Path("output") / args.run_name

    wandb_run = wandb.init(
        project="rug-locomotion-ppg", 
        config=args, 
        settings=wandb.Settings(start_method="fork"),
        group="sweep"
        )

    env_name = make_gym_env(args.env, args.data, visualize=args.visualize)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    learner = Learner(
        state_dim,
        action_dim,
        args.train_mode,
        args.ppo_delta,
        args.ppo_alpha,
        args.value_clip,
        args.entropy_coef,
        args.vf_loss_coef,
        args.ppo_batch_size,
        args.n_ppo_epochs,
        args.aux_batch_size,
        args.n_aux_epochs,
        args.beta_clone,
        args.gamma,
        args.lambd,
        args.learning_rate,
        args.initial_logstd
    )

    del env

    start = time.time()

    if (Path(output_path) / "agent.pth").exists():
        print("Another agent has initiated training with this configuration. Stopping...")
        comm.Abort()
        exit()

    learner.save_weights(output_path)

    msg = output_path
    output_path = comm.bcast(msg, root=0)

    try:
        avg_reward = 0
        avg_ep_time = 0
        data = None
        for epoch in infinite_range(0):

            data = comm.recv()
            trajectory, done_info = data

            states, actions, action_means, action_std, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, action_means, action_std, rewards, dones, next_states)

            learner.update_ppo()
            if epoch % args.n_aux_update == 0:
                learner.update_aux()

            learner.save_weights(output_path)

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
                                epoch,
                                time.time() - start,
                                avg_reward,
                                avg_ep_time
                            )
                        ),
                        step=real_epoch,
                    )

                print(
                    "Epoch {} (trajectory {}): reward {}, episode time {}".format(
                        real_epoch, epoch, avg_reward, avg_ep_time
                    )
                )

                if real_epoch == SWEEP_TIME:
                    break

                avg_reward = 0
                avg_ep_time = 0

    except Exception:
        print("Main terminated")
        print(traceback.format_exc())
    finally:
        if wandb_run:
            wandb_run.finish()
        finish = time.time()
        timedelta = finish - start
        print("Time: {}".format(str(datetime.timedelta(seconds=timedelta))))
        comm.Abort()


if __name__ == "__main__":
    args = get_args()

    args.num_workers = w_size

    if rank == 0:
        main_head(args)
    else:
        main_worker(args)
