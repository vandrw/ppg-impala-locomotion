from src.utils.logging import init_output
from pathlib import Path
import yaml
import wandb
import time
import numpy

SWEEP_ID = "" # Add your wandb sweep id here (i.e., user/project/sweep_id). 
SWEEP_RUNS = 25

sweep_path = init_output("sweep")
sweep = wandb.controller(SWEEP_ID)

for run in range(SWEEP_RUNS):
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

    with open(output_path / "config.yaml", "w") as f:
        yaml.safe_dump(dict_config, stream=f, default_flow_style=False)

    with open(sweep_path / "sweeps.info", "a") as f:
        f.write(str(output_path / "config.yaml") + '\n')
    