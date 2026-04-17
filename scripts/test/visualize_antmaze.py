import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def compute_episode_ranges(timeouts):
    """Split trajectories using timeouts only."""
    done_indices = np.where(timeouts.astype(bool))[0]

    if len(done_indices) == 0:
        return [(0, len(timeouts))]

    episode_starts = np.concatenate([[0], done_indices[:-1] + 1])
    episode_ranges = [(int(s), int(e) + 1) for s, e in zip(episode_starts, done_indices)]

    last_end = done_indices[-1] + 1
    if last_end < len(timeouts):
        episode_ranges.append((int(last_end), len(timeouts)))

    return episode_ranges


def find_success_segment(rewards, start_idx, end_idx, success_reward=1.0):
    """
    Return the first success index and the end of its consecutive success streak.

    Some AntMaze datasets keep reward=1 for several steps after reaching the goal
    before the environment resets. In that case we keep the whole consecutive
    success segment instead of cutting at the first hit.
    """
    reward_segment = rewards[start_idx:end_idx]
    success_mask = reward_segment >= success_reward
    success_indices = np.where(success_mask)[0]
    if len(success_indices) == 0:
        return None, None

    first_success_offset = int(success_indices[0])
    streak_end_offset = first_success_offset
    while (
        streak_end_offset + 1 < len(success_mask)
        and success_mask[streak_end_offset + 1]
    ):
        streak_end_offset += 1

    success_start_idx = start_idx + first_success_offset
    success_end_idx = start_idx + streak_end_offset + 1
    return int(success_start_idx), int(success_end_idx)


def load_xy_from_hdf5(h5file, use_qpos=False):
    """
    Load xy positions from an HDF5 dataset.

    Priority:
    - if use_qpos=True and infos/qpos exists: use infos/qpos[:, :2]
    - otherwise use observations[:, :2]
    """
    if use_qpos and "infos/qpos" in h5file:
        qpos = h5file["infos/qpos"][:]
        if qpos.ndim == 2 and qpos.shape[1] >= 2:
            return np.asarray(qpos[:, :2], dtype=np.float64), "infos/qpos[:, :2]"

    if "observations" in h5file:
        obs = h5file["observations"][:]
        if obs.ndim == 2 and obs.shape[1] >= 2:
            return np.asarray(obs[:, :2], dtype=np.float64), "observations[:, :2]"

    raise ValueError("Could not find usable xy positions in dataset.")


def load_antmaze_xy_from_hdf5(h5file, use_qpos=False, success_reward=1.0):
    xy, source = load_xy_from_hdf5(h5file, use_qpos=use_qpos)

    required_keys = ["rewards", "timeouts"]
    missing_keys = [key for key in required_keys if key not in h5file]
    if missing_keys:
        raise ValueError(
            "Dataset is missing keys required for success filtering: "
            + ", ".join(missing_keys)
        )

    rewards = np.asarray(h5file["rewards"][:]).reshape(-1)
    timeouts = np.asarray(h5file["timeouts"][:]).reshape(-1)

    if not (len(xy) == len(rewards) == len(timeouts)):
        raise ValueError(
            "Dataset arrays have inconsistent lengths: "
            f"xy={len(xy)}, rewards={len(rewards)}, timeouts={len(timeouts)}"
        )

    episode_ranges = compute_episode_ranges(timeouts)
    success_ranges = []
    success_hits = []
    for start_idx, end_idx in episode_ranges:
        success_start_idx, success_end_idx = find_success_segment(
            rewards,
            start_idx,
            end_idx,
            success_reward,
        )
        if success_start_idx is None:
            continue
        success_ranges.append((start_idx, success_end_idx))
        success_hits.append((success_start_idx, success_end_idx))

    all_xy = xy

    if success_ranges:
        success_xy = np.concatenate(
            [xy[start_idx:end_idx] for start_idx, end_idx in success_ranges],
            axis=0,
        )
        success_hit_xy = np.concatenate(
            [xy[start_idx:end_idx] for start_idx, end_idx in success_hits],
            axis=0,
        )
    else:
        success_xy = np.empty((0, 2), dtype=np.float64)
        success_hit_xy = np.empty((0, 2), dtype=np.float64)

    stats = {
        "num_transitions": int(len(xy)),
        "num_episodes": int(len(episode_ranges)),
        "num_successful_episodes": int(len(success_ranges)),
        "success_rate": float(len(success_ranges) / len(episode_ranges))
        if episode_ranges
        else 0.0,
        "num_all_points": int(all_xy.shape[0]),
        "num_success_points": int(success_xy.shape[0]),
        "num_success_hits": int(success_hit_xy.shape[0]),
    }

    return all_xy, success_xy, success_hit_xy, source, stats


def filter_valid_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] == 0:
        return xy.reshape(0, 2)

    mask = np.isfinite(xy).all(axis=1)
    xy = xy[mask]
    if xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    return xy


def sample_points(xy, max_points=None, seed=0):
    if max_points is None or xy.shape[0] <= max_points:
        return xy

    rng = np.random.default_rng(seed)
    indices = rng.choice(xy.shape[0], size=max_points, replace=False)
    return xy[indices]


def plot_scatter(
    all_xy,
    success_xy,
    success_hit_xy=None,
    title=None,
    output_path=None,
    figsize=(6.5, 6.0),
    dpi=300,
    marker="x",
    all_color="#5aa9e6",
    all_alpha=0.6,
    all_size=1.5,
    all_linewidths=1.25,
    success_color="#ff7a00",
    success_alpha=0.18,
    success_size=3.0,
    success_linewidths=0.5,
    hit_color="#ffd400",
    hit_alpha=0.15,
    hit_size=12.0,
    hit_linewidths=1.2,
    show_axes=True,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if all_xy.shape[0] > 0:
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=all_size,
            c=all_color,
            alpha=all_alpha,
            marker=marker,
            linewidths=all_linewidths,
            label="all episodes",
        )

    if success_xy.shape[0] > 0:
        ax.scatter(
            success_xy[:, 0],
            success_xy[:, 1],
            s=success_size,
            c=success_color,
            alpha=success_alpha,
            marker=marker,
            linewidths=success_linewidths,
            label="successful episodes",
        )

    if success_hit_xy is not None and success_hit_xy.shape[0] > 0:
        ax.scatter(
            success_hit_xy[:, 0],
            success_hit_xy[:, 1],
            s=hit_size,
            c=hit_color,
            alpha=hit_alpha,
            marker=marker,
            linewidths=hit_linewidths,
            label="success hits",
        )

    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title, fontsize=14)

    if show_axes:
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AntMaze dataset coverage with all episodes and successful episodes in different colors."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to a .hdf5 or .h5 file.",
    )
    parser.add_argument(
        "--use_qpos",
        action="store_true",
        help="Use infos/qpos[:, :2] instead of observations[:, :2]. Recommended for AntMaze.",
    )
    parser.add_argument(
        "--success_reward",
        type=float,
        default=1.0,
        help="Reward threshold used to decide whether an episode is successful.",
    )
    parser.add_argument(
        "--max_all_points",
        type=int,
        default=None,
        help="Optional max number of all-episode points to draw.",
    )
    parser.add_argument(
        "--max_success_points",
        type=int,
        default=None,
        help="Optional max number of successful-episode points to draw.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for point subsampling.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="antmaze_visualization.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--hide_axes",
        action="store_true",
        help="Hide axes for a cleaner figure.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    with h5py.File(args.dataset_path, "r") as h5file:
        all_xy, success_xy, success_hit_xy, source, stats = load_antmaze_xy_from_hdf5(
            h5file,
            use_qpos=args.use_qpos,
            success_reward=args.success_reward,
        )

    all_xy = filter_valid_xy(all_xy)
    success_xy = filter_valid_xy(success_xy)
    success_hit_xy = filter_valid_xy(success_hit_xy)

    all_xy = sample_points(all_xy, max_points=args.max_all_points, seed=args.seed)
    success_xy = sample_points(success_xy, max_points=args.max_success_points, seed=args.seed)
    success_hit_xy = sample_points(
        success_hit_xy,
        max_points=args.max_success_points,
        seed=args.seed,
    )

    print("AntMaze visualization summary:")
    print(f"Loaded xy from: {source}")
    print(f"Total transitions: {stats['num_transitions']}")
    print(f"Total timeout-defined episodes: {stats['num_episodes']}")
    print(f"Successful episodes: {stats['num_successful_episodes']}")
    print(f"Success rate: {stats['success_rate'] * 100:.2f}%")
    print(f"All-episode plotted points: {all_xy.shape[0]}")
    print(f"Successful-episode plotted points: {success_xy.shape[0]}")
    print(f"Success-hit points: {success_hit_xy.shape[0]}")
    if all_xy.shape[0] > 0:
        print(f"All-episode x range: [{all_xy[:, 0].min():.4f}, {all_xy[:, 0].max():.4f}]")
        print(f"All-episode y range: [{all_xy[:, 1].min():.4f}, {all_xy[:, 1].max():.4f}]")
    if success_xy.shape[0] > 0:
        print(
            f"Successful-episode x range: [{success_xy[:, 0].min():.4f}, {success_xy[:, 0].max():.4f}]"
        )
        print(
            f"Successful-episode y range: [{success_xy[:, 1].min():.4f}, {success_xy[:, 1].max():.4f}]"
        )

    title = args.title
    if title is None:
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        title = f"{dataset_name} all vs successful episodes"

    plot_scatter(
        all_xy=all_xy,
        success_xy=success_xy,
        success_hit_xy=success_hit_xy,
        title=title,
        output_path=args.output_path,
        show_axes=not args.hide_axes,
    )


if __name__ == "__main__":
    main()
