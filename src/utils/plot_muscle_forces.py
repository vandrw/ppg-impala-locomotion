import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import argparse

plt.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["Helvetica"],
        "font.size": 21,
        "lines.linewidth": 2,
    }
)

def main(args):
    forces = pd.read_csv(args.force_file, header=0)

    # from pathlib import Path
    # data_path = Path("final") / "forces_ab06.csv"
    # original_start = 14.2
    # original = pd.read_csv(data_path, header=0)
    # original_end = original_start + forces["time"].iloc[-1]
    # original = original[
    #         original["time"].between(
    #             original_start, original_start + args.end
    #         )
    #     ]

    if args.columns is None:
        args.columns = ["bifemsh", "vasti"]
    
    # The saving process on OpenSim will create the first two columns
    # as "time" and "Coordinate". Make sure you rename "time" to "step"
    # and "Coordinate" to "time".
    forces = forces.drop("step", axis=1)

    if args.start and args.end:
        forces = forces[forces["time"].between(args.start, args.end / args.speed)]
    elif args.start:
        forces = forces[forces["time"] >= args.start]
    elif args.end:
        forces = forces[forces["time"] <= args.end / args.speed]

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(len(args.columns), 2)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for j, side in enumerate(["r", "l"]):
        # Create subplot for each column and handle uneven number of columns
        for i, column in enumerate(args.columns):
            col = column + "_" + side
            
            if i == len(args.columns) - 1:
                ax = fig.add_subplot(gs[i, j - 1])
            else:
                ax = fig.add_subplot(gs[i, j - 1])
            
            # stdev = forces[col].std()

            # ax.plot(
            #     (original["time"] - original_start) / forces["time"].iloc[-1] / args.speed * 100,
            #     original[col],
            #     label="Imitation data",
            #     color="gray",
            #     alpha=0.3,
            # )
            # ax.fill_between(
            #     (original["time"] - original_start) / forces["time"].iloc[-1] / args.speed * 100,
            #     original[col] - stdev,
            #     original[col] + stdev,
            #     edgecolor="gray",
            #     facecolor="gray",
            #     alpha=0.3,
            # )
            # ax.axhline(original[col].mean(), color="black", linestyle="--", alpha=0.4)

            ax.plot(forces["time"] / forces["time"].iloc[-1] * 100, forces[col], label="Generated motion")
            ax.axhline(forces[col].mean(), c='r', linestyle='--')

            if side == "l":
                ax.set_title(col.split("_")[0].capitalize() + " Left")
            else:
                ax.set_title(col.split("_")[0].capitalize() + " Right")

            if j == 1:
                ax.set_ylabel("Fibre Force (N)")
            if i == len(args.columns) - 1:
                ax.set_xlabel("Gait cycle (%)")
                if j == 1:
                    ax.legend(bbox_to_anchor=(1.1, -0.57), loc="center", ncol=2)

    # plt.show()
    plt.savefig("forces.pdf", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "force_file",
        type=str,
        help="""Path to the force data file.
        You can obtain this through the plotting functionality of OpenSim,
        by going to Tools > Plot and selecting total-fiber force as the y-axis.
        This will create a .sto file, so you will have to convert it to a .csv.
        Make sure you also rename the column "time" to "step" and the column
        "Coordinate" to "time".""",
    )

    parser.add_argument(
        "--columns",
        type=str,
        nargs="*",
        help="Which columns you would like to plot. Default is all.",
    )

    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--speed", type=float, default=1)

    args = parser.parse_args()

    main(args)

