from src.utils.logging import init_output
from pathlib import Path
import yaml
import wandb
import time
import numpy
import argparse



def main(args):
    sweep_path = init_output("sweep")
    sweep = wandb.controller(args.id)

    print("Creating configs for sweep {}.".format(args.id))

    for run in range(args.runs):
        params = sweep.search()
        config = params.config
        dict_config = {}
        for key in config:
            if type(config[key]["value"]) == numpy.float64:
                dict_config[key] = float(config[key]["value"])
            else:
                dict_config[key] = config[key]["value"]
        
        run_name = "{}_{}".format(time.strftime("%Y%m%d%H%M%S"), run)
        output_path = init_output(Path("sweep") / run_name)
        dict_config["run_name"] = str(Path("sweep") / run_name)

        print("Created config in {}...".format(dict_config["run_name"]))
        with open(output_path / "config.yaml", "w") as f:
            yaml.safe_dump(dict_config, stream=f, default_flow_style=False)

        with open(sweep_path / "sweeps.info", "a") as f:
            f.write(str(output_path / "config.yaml") + '\n')
    
    print("Finished creating {} configs!".format(args.runs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Serach Initializer")

    parser.add_argument(
        "--id",
        type=str,
        help="The sweep id provided by wandb: <user>/<project>/<sweep id>"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="The number of sweep configurations you want to create."
    )

    args = parser.parse_args()
    main(args)
    