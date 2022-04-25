import argparse    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="The environment type used. Currently supports the healthy and prosthesis models.",
)
parser.add_argument(
    "--train",
    action="store_true",
    help="If you want to train the agent, set this to True. But set this otherwise if you only want to test it"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Display the environment. Disregard this argument if you run this on Peregrine/Google Collab."
)
parser.add_argument(
    "--n-update",
    type=int,
    default=1024,
    help="How many episodes before the Policy is updated."
)
parser.add_argument(
    "--n-aux-update",
    type=int,
    default=5,
    help="How many episodes before the Auxiliary is updated."
)
parser.add_argument(
    "--n-episode",
    type=int,
    default=1000000,
    help="How many episodes you want to run."
)
parser.add_argument(
    "--n-agent",
    type=int,
    default=4,
    help="How many agents you want to run asynchronously."
)
parser.add_argument(
    "--policy-kl-range",
    type=float,
    default=0.03,
    help="TODO."
)
parser.add_argument(
    "--policy-params",
    type=int,
    default=5,
    help="TODO."
)
parser.add_argument(
    "--value-clip",
    type=float,
    default=1.0,
    help="TODO."
)
parser.add_argument(
    "--entropy-coef",
    type=float,
    default=0.0,
    help="How much action randomness is introduced."
)
parser.add_argument(
    "--vf-loss-coef",
    type=float,
    default=1.0,
    help="How much action randomness is introduced."
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="How many batches per update. The number of batches is given by the number of updates divided by the batch size."
)
parser.add_argument(
    "--PPO-epochs",
    type=int,
    default=10,
    help="How many PPO epochs per update."
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="TODO."
)    
parser.add_argument(
    "--lam",
    type=float,
    default=0.95,
    help="TODO."
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2.5e-4,
    help="TODO."
)
parser.add_argument(
    "--project-name",
    type=str,
    help="Project name to use in wandb."
) 

args = parser.parse_args()
print(args)