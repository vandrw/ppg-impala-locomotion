from itertools import count as infinite_range
from src.utils.env_loader import make_gym_env
from src.ppg.agent import Agent
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

class Runner:
    def __init__(self, experiment_type, data, save_path):
        self.env = make_gym_env(experiment_type, data, visualize=False)
        self.states = self.env.reset()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = Agent(self.state_dim, self.action_dim, -1.34, False)

        self.max_action = 1.0

        self.save_path = save_path

    def get_motion(self, fps, tries):
        self.agent.load_weights(self.save_path)
        total_reward = 0
        
        best_pose = None
        best_rew = 0
        best_idx = 0

        for _ in infinite_range(0):
            action, _, _ = self.agent.act(self.states)
            next_state, _, done, _ = self.env.step(action)
            self.states = next_state
            
            if done:
                self.env.reset()
                break
    
        for t in range(tries):
            pose_info = pd.DataFrame()
            for frame in infinite_range(0):
                curr_pose = {}
                action, _, _ = self.agent.act(self.states)

                next_state, reward, done, info = self.env.step(action)

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
                    logging.info("{}. Reward: {}, Episode time: {}".format(t + 1, total_reward, frame))

                    if total_reward > best_rew:
                        best_pose = pose_info
                        best_rew = total_reward
                        best_idx = t + 1

                    total_reward = 0
                    self.env.reset()
                    break
        
        logging.info("Saving episode {} with reward {}".format(best_idx, best_rew))
        return best_pose

def save_episode(pose_info, output_path, output_name="episode", csv=True):
    if not csv:
        with (Path(output_path) / "{}.mot".format(output_name)).open("w") as file:
            header = "".join(("Coordinates\n",
                "version=1\n",
                "nRows={}\n".format(pose_info.shape[0]),
                "nColumns={}\n".format(pose_info.shape[1]),
                "inDegrees=no\n\n",
                "Units are S.I. units (second, meters, Newtons, ...)\n",
                "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n",
                "endheader\n"))

            file.write(header)

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
    else:
        with (Path(output_path) / "{}.csv".format(output_name)).open("w") as file:
            for col in pose_info.columns:
                if col == "time":
                    file.write("time")
                else:
                    file.write(",{}".format(col))

            file.write("\n")

            for _, row in pose_info.iterrows():
                for col in pose_info.columns:
                    if col == "time":
                        file.write("{:.8f}".format(row[col]))
                    else:
                        file.write(",{:.8f}".format(row[col]))
                file.write("\n")

def main(args):
    logging.info(
        "Creating motion with environment '{}' and starting data '{}' at {} FPS...".format(
            args.env, args.data, args.fps
        )
    )
    if args.tries > 1:
        logging.info("Performing {} tries...".format(args.tries))

    runner = Runner(args.env, args.data, args.folder_path)

    pose_info = runner.get_motion(args.fps, args.tries)

    save_episode(pose_info, args.folder_path, args.output, args.csv)

    if args.csv:
        ext = "csv"
    else:
        ext = "mot"

    logging.info("Motion saved at {}".format(str(Path(args.folder_path) / "{}.{}".format(args.output, ext))))

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

    parser.add_argument(
        "--csv",
        action="store_true",
        help="Whether to store the file as a '.csv'. This file cannot be imported in the OpenSim visualizer."
    )

    parser.add_argument(
        "--tries",
        type=int,
        default=1,
        help="How many tries to perform. If given more than one, the program will calculate multiple episodes and will return the best."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="episode",
        help="Name of the output file."
    )

    args = parser.parse_args()
    assert args.tries >= 1, "Please provide a value of one or higher!"

    main(args)
