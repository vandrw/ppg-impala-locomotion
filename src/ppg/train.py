import argparse
from mpi4py import MPI
from . import ppg
from . import torch_util as tu
from .impala_cnn import ImpalaEncoder
from . import logger
from .envs import get_venv

def train_fn(env_name="healthy",
    # distribution_mode="hard",
    arch="dual",  # 'shared', 'detach', or 'dual'
    # 'shared' = shared policy and value networks
    # 'dual' = separate policy and value networks
    # 'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference
    interacts_total=100_000_000,
    # num_envs=64,
    n_epoch_pi=1,
    n_epoch_vf=1,
    gamma=.999,
    aux_lr=5e-4,
    lr=5e-4,
    nminibatch=8,
    aux_mbsize=4,
    clip_param=.2,
    kl_penalty=0.0,
    n_aux_epochs=6,
    n_pi=32,
    beta_clone=1.0,
    vf_true_weight=1.0,
    log_dir='/tmp/ppg',
    comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    tu.setup_dist(comm=comm)
    tu.register_distributions_for_tree_util()

    if log_dir is not None:
        format_strs = ['csv', 'stdout'] if comm.Get_rank() == 0 else []
        logger.configure(comm=comm, dir=log_dir, format_strs=format_strs)

    venv = get_venv(name=env_name)

    enc_fn = lambda obtype: ImpalaEncoder(
        obtype.shape,
        outsize=256,
        chans=(116,),
    )
    model = ppg.PhasicValueModel(venv.observation_space, venv.action_space, arch=arch)

    model.to(tu.dev())
    logger.log(tu.format_model(model))
    tu.sync_params(model.parameters())

    name2coef = {"pol_distance": beta_clone, "vf_true": vf_true_weight}

    ppg.learn(
        venv=venv,
        model=model,
        interacts_total=interacts_total,
        ppo_hps=dict(
            lr=lr,
            γ=gamma,
            λ=0.95,
            nminibatch=nminibatch,
            n_epoch_vf=n_epoch_vf,
            n_epoch_pi=n_epoch_pi,
            clip_param=clip_param,
            kl_penalty=kl_penalty,
            log_save_opts={"save_mode": "last"}
        ),
        aux_lr=aux_lr,
        aux_mbsize=aux_mbsize,
        n_aux_epochs=n_aux_epochs,
        n_pi=n_pi,
        name2coef=name2coef,
        comm=comm,
    )

def main():
    parser = argparse.ArgumentParser(description='Process PPG training arguments.')
    parser.add_argument('--env-name', type=str, default='coinrun')
    # parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--n-epoch-pi', type=int, default=1)
    parser.add_argument('--n-epoch-vf', type=int, default=1)
    parser.add_argument('--n-aux-epochs', type=int, default=6)
    parser.add_argument('--n-pi', type=int, default=32)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--kl-penalty', type=float, default=0.0)
    parser.add_argument('--arch', type=str, default='dual') # 'shared', 'detach', or 'dual'

    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    train_fn(
        env_name=args.env_name,
        # num_envs=args.num_envs,
        n_epoch_pi=args.n_epoch_pi,
        n_epoch_vf=args.n_epoch_vf,
        n_aux_epochs=args.n_aux_epochs,
        n_pi=args.n_pi,
        arch=args.arch,
        comm=comm)

if __name__ == '__main__':
    main()
