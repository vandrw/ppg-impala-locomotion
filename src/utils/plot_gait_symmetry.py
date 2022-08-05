import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import argparse
from math import pi
from opensim_env.data.core import ANGULAR_COORDINATE_NAMES

plt.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["Helvetica"],
        "font.size": 21,
        "lines.linewidth": 3,
    }
)


def main(args):
    original_start = 0
    if args.subject == "AB06":
        data_path = Path("data") / "AB06_transformed_inDegrees_14,2.csv"
        original_start = 14.2
    elif args.subject == "AB23":
        data_path = Path("data") / "AB23_transformed_inDegrees_6,52.csv"
        original_start = 6.52
    else:
        raise ValueError("The provided subject name does not exist.")

    original = pd.read_csv(data_path, header=0)
    result = pd.read_csv(args.motion_path, header=0)

    original_end = original_start + result["time"].iloc[-1]

    if args.start and args.end:
        result = result[result["time"].between(args.start, args.end / args.speed)]
        original = original[
            original["time"].between(
                original_start + args.start, original_start + args.end
            )
        ]
    elif args.start:
        result = result[result["time"] >= args.start]
        original = original[
            original["time"].between(original_start + args.start, original_end)
        ]
    elif args.end:
        result = result[result["time"] <= args.end / args.speed]
        original = original[
            original["time"].between(original_start, original_start + args.end)
        ]
    else:
        original = original[original["time"].between(original_start, original_end)]

    if args.columns is None:
        args.columns = ["hip_adduction", "knee_angle", "ankle_angle"]

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(len(args.columns), 2)
    plt.subplots_adjust(hspace=0.5)

    for j, side in enumerate(["r", "l"]):
        # Create subplot for each column and handle uneven number of columns
        for i, column in enumerate(args.columns):
            col = column + "_" + side
            if col in ANGULAR_COORDINATE_NAMES:
                result[col] *= 180 / pi
                stdev = 10
            else:
                # The data is in milimeters.
                stdev = 0.2

            if i == len(args.columns) - 1:
                ax = fig.add_subplot(gs[i, j - 1])
            else:
                ax = fig.add_subplot(gs[i, j - 1])

            ax.plot(
                (original["time"] - original_start) / result["time"].iloc[-1] / args.speed * 100,
                original[col],
                label="Imitation data",
                color="gray",
                alpha=0.3,
            )
            ax.fill_between(
                (original["time"] - original_start) / result["time"].iloc[-1] / args.speed * 100,
                original[col] - stdev,
                original[col] + stdev,
                edgecolor="gray",
                facecolor="gray",
                alpha=0.3,
            )

            ax.plot(result["time"] / result["time"].iloc[-1] * 100, result[col], label="Generated motion")

            if side == "l":
                ax.set_title(col.split("_")[0].capitalize() + " Left")
            else:
                ax.set_title(col.split("_")[0].capitalize() + " Right")

            if j == 1:
                ax.set_ylabel("Angle (degrees)")
            if i == len(args.columns) - 1:
                ax.set_xlabel("Gait cycle (%)")
                if j == 1:
                    ax.legend(bbox_to_anchor=(1.1, -0.57), loc="center", ncol=2)

    # plt.show()
    plt.savefig("symmetry.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "motion_path", type=str, help="Path to the obtained motion data file."
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        help="""Which columns you would like to plot. 
        Default is hip_adduction, knee_angle and ankle_angle.
        The program will automatically plot both legs.""",
    )
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--speed", type=float, default=1)

    parser.add_argument("--subject", type=str, default="AB06", help="The subject name.")

    args = parser.parse_args()

    main(args)
