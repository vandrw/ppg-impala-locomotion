from itertools import count as infinite_range
from src.utils.env_loader import make_gym_env
from src.ppg.agent import Agent
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import gym


class Runner:
    def __init__(self, experiment_type, data, save_path):
        env_name = make_gym_env(experiment_type, data, visualize=False)
        self.env = gym.make(env_name)
        self.states = self.env.reset()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = Agent(self.state_dim, self.action_dim, True)

        self.max_action = 1.0

        self.save_path = save_path

    def get_motion(self, fps):
        self.agent.load_weights(self.save_path)
        total_reward = 0
        pose_info = pd.DataFrame()

        for frame in infinite_range(0):
            curr_pose = {}
            action, _ = self.agent.act(self.states)

            action_gym = np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, done, info = self.env.step(action_gym)

            pose = asdict(info["obs/dyn_pose"])
            curr_pose["time"] = frame / fps
            for body in pose["pose"]:
                curr_pose[body] = pose["pose"][body]
            curr_pose["lumbar_extension"] = -5.00000214

            if pose_info.empty:
                pose_info = pd.DataFrame(curr_pose, index=[0])
            else:
                pose_info.loc[-1] = curr_pose.values()
                pose_info.index += 1

            total_reward += reward
            self.states = next_state

            if done:
                print("Reward: {}, Episode time: {}".format(total_reward, frame))

                return pose_info


def main(args):
    print(
        "Creating motion with environment '{}' and starting data '{}' at {} FPS...".format(
            args.env, args.data, args.fps
        )
    )

    runner = Runner(args.env, args.data, args.folder_path)

    pose_info = runner.get_motion(args.fps)
    with (Path(args.folder_path) / "episode.mot").open("w") as file:
        file.write(
            """Coordinates
version=1
nRows={}
nColumns={}
inDegrees=no

Units are S.I. units (second, meters, Newtons, ...)
If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).

endheader
""".format(
                pose_info.shape[0], pose_info.shape[1]
            )
        )

        for col in pose_info.columns:
            if col == "time":
                file.write("time")
            else:
                file.write("\t{}".format(col))

        file.write("\n")

        for _, row in pose_info.iterrows():
            for col in pose_info.columns:
                if col == "time":
                    file.write("      {:.8f}".format(row[col]))
                else:
                    file.write("\t     {:.8f}".format(row[col]))
            file.write("\n")

    print("Motion saved at {}".format(str(Path(args.folder_path) / "episode.mot")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder of the model checkpoint. Only the folder!",
    )
    parser.add_argument(
        "--env",
        help="The environment type: 'healthy' or 'healthy_terrain'.",
        default="healthy",
    )

    parser.add_argument(
        "--data",
        help="The participant whose data was used: 'AB06' or 'AB23'.",
        default="AB06",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=200,
        help="How many frames per second the result will have.",
    )

    args = parser.parse_args()

    main(args)
