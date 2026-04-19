"""Render a single BEV frame per typical scene category.

Picks one representative scene per category (highway / intersection / dense /
sparse) from the already-classified feature pkls, then uses trajdata's
UnifiedDataset + `plot_scene_batch` to render a BEV snapshot with vehicle
bounding boxes and optional history/future trajectories.

Outputs (in --output-dir):
    scene_highway.png
    scene_intersection.png
    scene_dense.png
    scene_sparse.png
    typical_scenes_grid.png  (2x2 composite)

Usage:
    python -m MaxEntIRL.render_typical_scenes
    python -m MaxEntIRL.render_typical_scenes \\
        --nuscenes-path ~/datasets/nuScenes --data-split nusc_trainval-val
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from trajdata import AgentType, UnifiedDataset
from trajdata.visualization.vis import plot_scene_batch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from evaluate_weight_network import (  # noqa: E402
    CATEGORIES,
    CATEGORY_LABELS,
    classify_all_scenes,
    select_representatives,
)


def build_dataset(data_split: str, nuscenes_path: str,
                  history_sec: float, future_sec: float) -> UnifiedDataset:
    env_key = data_split.split("-")[0]  # e.g. "nusc_trainval"
    return UnifiedDataset(
        desired_data=[data_split],
        centric="scene",
        desired_dt=0.1,
        history_sec=(history_sec, history_sec),
        future_sec=(future_sec, future_sec),
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 400,
            "offset_frac_xy": (0.0, 0.0),
            "return_rgb": True,
        },
        max_agent_num=20,
        num_workers=0,
        verbose=True,
        data_dirs={env_key: nuscenes_path},
    )


def find_batches_for_scenes(dataset: UnifiedDataset, wanted_scene_ids: set):
    """Iterate the dataset until we have one batch per wanted scene_id."""
    collate_fn = dataset.get_collate_fn()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    found = {}  # scene_id -> SceneBatch
    wanted = set(wanted_scene_ids)
    for batch in loader:
        if not wanted:
            break
        sids = batch.scene_ids if batch.scene_ids else []
        for sid in sids:
            sid_str = str(sid)
            for target in list(wanted):
                if target == sid_str or target in sid_str or sid_str in target:
                    if target not in found:
                        found[target] = batch
                        wanted.discard(target)
                        print(f"  matched scene_id '{sid_str}' for '{target}'")
                        break
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        default=os.path.join(_SCRIPT_DIR, "irl_output", "features"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_SCRIPT_DIR, "irl_output"),
    )
    parser.add_argument(
        "--nuscenes-path",
        default="../behavior-generation-dataset/nuscenes",
        help="Path to nuScenes dataset (contains maps/, samples/, sweeps/, ...)",
    )
    parser.add_argument(
        "--data-split",
        default="nusc_trainval-val",
        help="trajdata split tag (e.g. nusc_trainval-val, nusc_mini-mini_val)",
    )
    parser.add_argument("--history-sec", type=float, default=1.0)
    parser.add_argument("--future-sec", type=float, default=2.0)
    parser.add_argument(
        "--no-traj",
        action="store_true",
        help="Disable history/future trajectories (only boxes).",
    )
    args = parser.parse_args()

    if args.no_traj:
        args.history_sec = 0.1  # minimal, need at least current step
        args.future_sec = 0.0

    # Step 1: classify pkl files and pick 4 representatives.
    print(f"Classifying scenes in {args.feature_dir}")
    classified = classify_all_scenes(args.feature_dir)
    for cat in CATEGORIES:
        print(f"  {CATEGORY_LABELS[cat]:26s}: {len(classified[cat])} scenes")

    reps = select_representatives(classified)
    if not reps:
        print("Error: no representatives found. Feature dir empty?")
        return
    print("\nRepresentatives:")
    rep_name_to_cat = {}
    for cat, (scene_name, _) in reps.items():
        print(f"  {CATEGORY_LABELS[cat]:26s} -> {scene_name}")
        rep_name_to_cat[scene_name] = cat

    # Step 2: build trajdata dataset.
    print(f"\nBuilding UnifiedDataset ({args.data_split})")
    dataset = build_dataset(
        args.data_split, args.nuscenes_path,
        args.history_sec, args.future_sec,
    )
    print(f"  dataset size: {len(dataset):,}")

    # Step 3: find a SceneBatch per representative.
    print("\nSearching for matching batches")
    found = find_batches_for_scenes(dataset, set(rep_name_to_cat.keys()))

    missing = set(rep_name_to_cat.keys()) - set(found.keys())
    if missing:
        print(f"  warning: no match found for {missing}")

    # Step 4: render one PNG per category.
    os.makedirs(args.output_dir, exist_ok=True)
    rendered = {}
    for scene_name, batch in found.items():
        cat = rep_name_to_cat[scene_name]
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_scene_batch(batch, batch_idx=0, ax=ax,
                         legend=False, show=False, close=False)
        ax.set_title(f"{CATEGORY_LABELS[cat]}\n({scene_name})",
                     fontsize=12, fontweight="bold")
        out_path = os.path.join(args.output_dir, f"scene_{cat}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rendered[cat] = out_path
        print(f"  saved {out_path}")

    # Step 5: stitch 2x2 grid.
    if rendered:
        order = ["highway", "intersection", "dense", "sparse"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
        for ax, cat in zip(axes.flatten(), order):
            ax.axis("off")
            if cat in rendered:
                ax.imshow(mpimg.imread(rendered[cat]))
        fig.tight_layout()
        grid_path = os.path.join(args.output_dir, "typical_scenes_grid.png")
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {grid_path}")


if __name__ == "__main__":
    main()
