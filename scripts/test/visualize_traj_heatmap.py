import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_xy_from_hdf5(h5file):
    # AntMaze / Mujoco 类环境优先用 qpos 的前两维
    if "infos/qpos" in h5file:
        qpos = h5file["infos/qpos"][:]
        if qpos.ndim == 2 and qpos.shape[1] >= 2:
            return np.asarray(qpos[:, :2], dtype=np.float64), "infos/qpos[:, :2]"

    # Maze2D 常见情况：observations 前两维就是 (x, y)
    if "observations" in h5file:
        obs = h5file["observations"][:]
        if obs.ndim == 2 and obs.shape[1] >= 2:
            return np.asarray(obs[:, :2], dtype=np.float64), "observations[:, :2]"

    raise ValueError("Could not find usable xy positions in HDF5 file.")


def filter_valid_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    mask = np.isfinite(xy).all(axis=1)
    xy = xy[mask]
    if xy.shape[0] == 0:
        raise ValueError("No valid finite xy points found.")
    return xy


def plot_density_heatmap(
    xy,
    bins=120,
    figsize=(7, 7),
    dpi=300,
    cmap="magma",
    alpha=0.9,
    show_points=False,
    point_size=0.2,
    point_alpha=0.08,
    title=None,
    output_path=None,
):
    x = xy[:, 0]
    y = xy[:, 1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    hist = ax.hist2d(x, y, bins=bins, cmap=cmap)

    cbar = plt.colorbar(hist[3], ax=ax)
    cbar.set_label("Visit count")

    if show_points:
        ax.scatter(x, y, s=point_size, alpha=point_alpha)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 2D visit-density heatmap from an HDF5 dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to a .hdf5 or .h5 dataset file.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of bins per axis for the heatmap.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the figure.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="heatmap.png",
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--show_points",
        action="store_true",
        help="Overlay raw points as a faint scatter layer.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    with h5py.File(args.dataset_path, "r") as f:
        xy, source = load_xy_from_hdf5(f)

    xy = filter_valid_xy(xy)

    print(f"Loaded {xy.shape[0]} xy points from: {source}")
    print(f"x range: [{xy[:, 0].min():.4f}, {xy[:, 0].max():.4f}]")
    print(f"y range: [{xy[:, 1].min():.4f}, {xy[:, 1].max():.4f}]")

    title = args.title
    if title is None:
        title = f"Visit Density Heatmap\n{os.path.basename(args.dataset_path)}"

    plot_density_heatmap(
        xy=xy,
        bins=args.bins,
        title=title,
        output_path=args.output_path,
        show_points=args.show_points,
    )

    plt.show()


if __name__ == "__main__":
    main()