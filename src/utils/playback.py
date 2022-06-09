# Playback script written by Robin Kock.

from argparse import ArgumentParser
from pathlib import Path
from time import time, sleep
import json
from typing import List
import re
from tempfile import TemporaryDirectory

from opensim import Model, Manager
import pandas as pd

from opensim_env.data import TrainingData
from opensim_env.models import HEALTHY_PATH


def fps_loop(fps):
    start = time()
    frame_index = 0
    frame_time = 0
    while True:
        yield frame_time

        frame_index += 1
        frame_time = frame_index / fps
        next_in = frame_time - (time() - start)

        if next_in > 0.0:
            sleep(next_in)
        elif -next_in > 1 / fps:
            drop_count = max(int(-next_in / fps), 1)
            frame_index += drop_count
            frame_time = frame_index / fps
            print("Dropped", drop_count, "frames")
        elif int(frame_index) % 5 == 0:
            print("Frames lagging behind", -next_in, "seconds")


WANDB_URL_MATCHER = re.compile(
    "^(.+wandb.ai/)?(?P<org>[^/]+)/(?P<proj>[^/]+)(/runs)?/(?P<run>[^/]+)([\\?#].+)?$"
)


if __name__ == "__main__":
    # fmt: off
    parser = ArgumentParser("playback")

    parser.add_argument("data", type=str,
                    help="Path to data file or wandb path/url.")

    parser.add_argument("--model", type=Path, default=HEALTHY_PATH,
                    help="Path to model.")

    parser.add_argument("--fps", type=float, default=15,
                    help="How often to update the visualizer.")

    parser.add_argument("--speed", type=float, default=1,
                    help="Mutliply the playback speed.")

    parser.add_argument("--deg-to-rad", action="store_true",
                    help="Convert data from degrees to radians.")

    parser.add_argument("--repeat", action="store_true",
                    help="Repeat data until window is closed.")

    parser.add_argument("--start", type=float, default=0,
                    help="Start after the given amount of seconds.")

    parser.add_argument("--wandb", action="store_true", 
                    help="Download a run from a weights and biases table artifact.")

    parser.add_argument("--filter", nargs=1, action="append", type=str,
                    help="Filter wandb artifacts, must include FILTER, can be "
                         "specified multiple times")
    # fmt: on

    args = parser.parse_args()

    if args.wandb:
        mg = WANDB_URL_MATCHER.match(args.data).group
        wandb_path = f"{mg('org')}/{mg('proj')}/{mg('run')}"
        import wandb

        api = wandb.Api()
        run = api.run(wandb_path)
        artifact_tables = [
            a
            for a in iter(run.logged_artifacts())
            if a.type == "run_table" and all((f[0] in a.name) for f in args.filter)
        ]

        assert len(artifact_tables) != 0

        if len(artifact_tables) == 1:
            artifact = artifact_tables[0]
        else:
            for i, a in enumerate(artifact_tables):
                print(f"{i:4}: {a.name}")

            choice = input("Choose an artifact to view (empty for last): ").strip()
            artifact_idx = -1 if choice == "" else int(choice)
            artifact = artifact_tables[artifact_idx]

        with TemporaryDirectory(prefix="opensim-playback-") as tmp:
            artifact_path = Path(artifact.file(root=tmp))
            json_table = json.loads(artifact_path.read_text())

        columns: List[str] = json_table["columns"]

        data_or_path = pd.DataFrame(json_table["data"], columns=columns)
        for col in columns:
            if col.startswith("pose."):
                data_or_path.rename(columns={col: col[5:]}, inplace=True)
            elif col.startswith("velocity."):
                del data_or_path[col]
            else:
                raise Exception()
        data_or_path["time"] = [i / 50.0 for i in range(len(data_or_path.index))]
    else:
        data_or_path = args.data

    data = TrainingData(
        data_or_path,
        deg_to_rad=args.deg_to_rad,
        tempo=args.speed,
        start_time=args.start,
    )

    model = Model(str(args.model))
    model.setUseVisualizer(True)
    state = model.initSystem()

    state = model.initializeState()

    time_offset = 0

    for t_real in fps_loop(args.fps):
        t = t_real - time_offset
        pose = data.get_row(t)

        if pose is None:
            if t < 0.01:
                print("WARNING: start time is likely invalid")
                break
            elif args.repeat:
                time_offset = t_real
                continue
            else:
                break

        for coordinate in model.getCoordinateSet():
            if coordinate.get_locked():
                continue
            name = coordinate.getName()
            coordinate.setValue(state, getattr(pose.pose, name))
            coordinate.setSpeedValue(state, getattr(pose.velocity, name))

        try:
            # model.equilibrateMuscles(state)
            state.setTime(t)

            manager = Manager(model)
            manager.setIntegratorAccuracy(0.01)
            manager.initialize(state)


            manager.integrate(t + 0.0001)
        except RuntimeError as e:
            if "Resource deadlock avoided" in str(e):
                raise e
            print("Got an opensim Error: ", e)

    del model

    print("Done")
