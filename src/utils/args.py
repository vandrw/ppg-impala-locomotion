import argparse
import yaml


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


def get_args():
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

    # OpenSim
    parser.add_argument(
        "--env",
        help="The environment type used. Which model to import: 'healthy' or 'healthy_terrain'.",
    )
    parser.add_argument(
        "--data",
        help="The dataset used. Currently supports the 'AB06' and 'AB23' subjects.",
    )

    # Logging
    parser.add_argument(
        "--run_name",
        type=str,
        help="Display name to use in wandb. Also used for the path to save the model. Provide a unique name, or give the same name to continue training.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the environment. Disregard this argument if you run this on Peregrine/Google Collab.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to save output on wandb."
    )

    # Worker
    parser.add_argument(
        "--train_mode",
        action="store_true",
        help="Whether to save new checkpoints during learning. If used, there will be changes made to the model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="How many agents you want to run asynchronously. Only used for the Ray implementation.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1024,
        help="How many steps to perform in the environment before the networks are updated (per worker).",
    )
    parser.add_argument(
        "--normalize_obs",
        action="store_true",
        help="Whether to normalize the observations.",
    )
    parser.add_argument(
        "--obs_clip_range",
        type=float,
        default=5.0,
        help="Indicates how much the observations will be clipped after normalization.",
    )

    # TrulyPPO
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=5,
        help="For how many epochs to update the PPO networks (both value and policy).",
    )
    parser.add_argument(
        "--ppo_batchsize",
        type=int,
        default=16,
        help="Indicates how many batches will be used per update. The number of batches is equal to n_steps / batch_size.",
    )
    parser.add_argument(
        "--ppo_kl_range",
        type=float,
        default=0.05,
        help="Decides the amount of clipping, indicating the (KL) trust region for the policy.",
    )
    parser.add_argument(
        "--slope_rollback",
        type=float,
        default=5.0,
        help="Decides the force of the rollback.",
    )
    parser.add_argument(
        "--slope_likelihood",
        type=float,
        default=1.0,
        help="Decides whether we take into account the likelihood ratio between the old and the " 
        + "new policy when performing the rollback. It should be either 0 or 1, but it is possible "
        + "to provide any value inside or outside this interval. If this value is 0, try to provide "
        + "a smaller value for slope_rollback (preferably < 1.0)."
    )

    # Critic (PPO)
    parser.add_argument(
        "--val_clip_range",
        type=float,
        default=1.0,
        help="How much the critic values will be clipped, resulting in predicted values in the interval [-val_clip_range, val_clip_range].",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.0,
        help="How much action randomness is introduced.",
    )
    parser.add_argument(
        "--vf_loss_coef",
        type=float,
        default=1.0,
        help="Value function coefficient. Indicates how much the critic loss is taken into account.",
    )

    # Auxiliary
    parser.add_argument(
        "--aux_update",
        type=int,
        default=16,
        help="After how many sets of trajectories the Auxiliary is updated.",
    )
    parser.add_argument(
        "--aux_epochs",
        type=int,
        default=6,
        help="For how many epochs to train the Auxiliary policy.",
    )
    parser.add_argument(
        "--aux_batchsize",
        type=int,
        default=32,
        help="The size of each minibatches, per auxiliary epoch.",
    )
    parser.add_argument(
        "--beta_clone",
        type=float,
        default=1.0,
        help="Controls the trade-off between the old and new Auxiliary policy.",
    )

    # Optimization
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor. Reduces the value of future states.",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=0.95,
        help="GAE parameter used to reduce variance during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-4,
        help="Indicates the step-size taken during gradient descent.",
    )
    parser.add_argument(
        "--initial_logstd",
        type=float,
        default=-1.34,
        help="Decides the initial value of the log variance of the policy distribution.",
    )

    args = _parse_args(parser, config_parser)

    assert (
        args.env is not None
    ), "Provide an environment type: 'healthy', 'prosthesis', 'terrain' "

    if args.run_name is None:
        import time

        args.run_name = "{}_{}".format(args.env, time.strftime("%Y%m%d%H%M%S"))

    return args
