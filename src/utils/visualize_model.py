from src.ppg.runner import Runner
import argparse

def main(args):
    print(
        "Visualizing '{}' model with starting data '{}'...".format(
            args.env, args.data
        )
    )

    runner = Runner(
        args.env,
        args.data,
        -1.34,
        True,
        True,
        1000,
        0,
        save_path=args.folder_path,
    )

    runner.run_episode(0, 0, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSim Model Visualizer")

    parser.add_argument(
        "folder_path",
        help="Path to the folder of the trained model. Only the folder!"
        )
    parser.add_argument(
        "--env",
        help="The environment type: 'healthy' or 'healthy_terrain'.",
        default="healthy"
    )

    parser.add_argument(
        "--data",
        help="The participant whose data was used: 'AB06' or 'AB23'.",
        default="AB06"
    )

    args = parser.parse_args()

    main(args)