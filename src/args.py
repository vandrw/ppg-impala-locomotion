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
    parser.add_argument(
        "--env",
        help="The environment type used. Currently supports the healthy and prosthesis models.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Display name to use in wandb. Also used for the path to save the model. Provide a unique name.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to save output on wandb."
    )
    parser.add_argument(
        "--train_mode",
        action="store_true",
        help="Whether to save new checkpoints during learning. If used, there will be changes made to the model.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display the environment. Disregard this argument if you run this on Peregrine/Google Collab.",
    )
    parser.add_argument(
        "--n_update",
        type=int,
        default=1024,
        help="How many episodes before the Policy is updated. Also regarded as the simulation budget (steps) per iteration.",
    )
    parser.add_argument(
        "--n_aux_update",
        type=int,
        default=5,
        help="How many episodes before the Auxiliary is updated.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="How many agents you want to run asynchronously. Only used for the Ray implementation.",
    )
    parser.add_argument("--policy_kl_range", type=float, default=0.03, help="TODO.")
    parser.add_argument("--policy_params", type=int, default=5, help="TODO.")
    parser.add_argument("--value_clip", type=float, default=1.0, help="TODO.")
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.0,
        help="How much action randomness is introduced.",
    )
    parser.add_argument("--vf_loss_coef", type=float, default=1.0, help="TODO.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="How many batches per update. The number of batches is given by the number of updates divided by the batch size.",
    )
    parser.add_argument(
        "--PPO_epochs", type=int, default=10, help="How many PPO epochs per update."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="TODO.")
    parser.add_argument("--lam", type=float, default=0.95, help="TODO.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="TODO.")

    args = _parse_args(parser, config_parser)

    assert (
        args.env is not None
    ), "Provide an environment type: 'healthy', 'prosthesis', 'terrain' "

    if args.run_name is None:
        import time
        args.run_name = "{}_{}".format(args.env, time.strftime("%Y%m%d%H%M%S"))
        print("No unique run name was provided. Automatically named this run '{}'.".format(args.run_name))

    return args