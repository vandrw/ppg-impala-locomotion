from dataclasses import dataclass
from typing import Dict

from pathlib import Path
import sys

import yaml
import logging
import wandb

@dataclass
class DoneInfo:
    reward: float
    episode_len: float
    distance: float
    reward_partials: Dict[str, float]
@dataclass
class EpochInfo:
    epoch: int
    time: float
    info: DoneInfo

def init_output(run_name):
    output_path = Path("output") / run_name

    if not output_path.exists():
        Path.mkdir(output_path)

    return output_path

def init_logging(config):

    output_path = init_output(config.run_name)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

    logging.info(
        "Saving configuration in {}/{}.".format(output_path, "config.yaml")
    )
    with open(Path(output_path) / "config.yaml", "w") as f:
        f.write(yaml.safe_dump(config.__dict__, default_flow_style=False))

    continue_run = (Path(output_path) / "agent.pth").exists()
    if continue_run:
        try:
            with open(Path(output_path) / "epoch.info", "r") as ep_file:
                start_epoch = int(ep_file.readline())
            logging.info(
                "Found previous model in {}. Continuing training from epoch {}.".format(
                    output_path, start_epoch
                )
            )
        except FileNotFoundError:
            start_epoch = 0
    else:
        start_epoch = 0

    if config.log_wandb:
        try:
            wandb_run = wandb.init(
                project="rug-locomotion-ppg",
                config=config,
                name=config.run_name,
                id=config.run_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )
        except ModuleNotFoundError:
            logging.error(
                "You've requested to log metrics to wandb but package was not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )
            wandb_run = None
    
    return wandb_run, continue_run, start_epoch, output_path