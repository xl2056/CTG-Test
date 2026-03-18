#!/usr/bin/env python3
"""
Visualize CTG diffusion rollouts with learned IRL reward guidance.

Complete pipeline:
  Step 1: Train CTG diffusion model           → checkpoint (.ckpt)
  Step 2: Extract IRL features                → features data
  Step 3: Train MaxEntIRL reward              → reward weights (.pkl)
  Step 4: Run this script                     → guided rollout videos (.mp4)

Usage examples:

  # Basic: run with default config (singapore, 10 scenes)
  python run_visualize.py

  # Custom IRL weights and location
  python run_visualize.py --pkl_path ./MaxEntIRL/irl_output/weights/boston_99.pkl \
                          --location boston

  # Render only first frames (faster)
  python run_visualize.py --render_img --no-render_video

  # Custom checkpoint
  python run_visualize.py --ckpt_dir ./CTG/diffuser_trained_models/test/run0 \
                          --ckpt_key iter100000.ckpt

  # More scenes, different draw mode
  python run_visualize.py --num_scenes 20 --draw_mode entire_traj

  # Adjust guidance weight (higher = IRL reward matters more)
  python run_visualize.py --guidance_weight 5.0
"""

import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CTG diffusion rollouts with learned IRL reward guidance and generate videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # === Required paths ===
    parser.add_argument("--pkl_path", type=str, default=None,
                        help="Path to IRL weights .pkl file. "
                             "Default: ./MaxEntIRL/irl_output/weights/{location}_99.pkl")
    parser.add_argument("--ckpt_dir", type=str, default="./CTG/diffuser_trained_models/test/run0",
                        help="CTG diffusion model checkpoint directory")
    parser.add_argument("--ckpt_key", type=str, default="iter100000.ckpt",
                        help="Checkpoint key (filename)")

    # === Scene settings ===
    parser.add_argument("--location", type=str, default="singapore",
                        choices=["boston", "singapore"],
                        help="Scene location filter")
    parser.add_argument("--num_scenes", type=int, default=10,
                        help="Number of scenes to evaluate")
    parser.add_argument("--num_scenes_per_batch", type=int, default=1,
                        help="Batch size for scene processing")

    # === Guidance ===
    parser.add_argument("--guidance_weight", type=float, default=1.0,
                        help="Weight for IRL reward guidance (higher = stronger guidance)")
    parser.add_argument("--no_guidance", action="store_true",
                        help="Disable IRL guidance (baseline rollout)")

    # === Rendering ===
    parser.add_argument("--render_video", action="store_true", default=True,
                        help="Generate MP4 videos (default: True)")
    parser.add_argument("--no-render_video", dest="render_video", action="store_false",
                        help="Disable video rendering")
    parser.add_argument("--render_img", action="store_true", default=False,
                        help="Render only first frame images")
    parser.add_argument("--draw_mode", type=str, default="action",
                        choices=["action", "entire_traj", "map"],
                        help="Visualization draw mode")
    parser.add_argument("--render_size", type=int, default=400,
                        help="Render image size in pixels")
    parser.add_argument("--px_per_m", type=float, default=2.0,
                        help="Pixels per meter for rendering")
    parser.add_argument("--save_every_n_frames", type=int, default=5,
                        help="Save every N-th frame to video")

    # === Output ===
    parser.add_argument("--output_dir", type=str, default="./DiffusionGan/results",
                        help="Output directory for results and videos")

    # === Advanced ===
    parser.add_argument("--horizon", type=int, default=50,
                        help="Rollout horizon (number of future steps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve pkl path
    if args.pkl_path is None:
        args.pkl_path = f"./MaxEntIRL/irl_output/weights/{args.location}_99.pkl"

    # Validate paths
    if not os.path.isdir(args.ckpt_dir):
        print(f"[ERROR] Checkpoint directory not found: {args.ckpt_dir}")
        print("  You need to train the CTG diffusion model first.")
        print("  See CTG/scripts/ for training scripts.")
        sys.exit(1)

    if not args.no_guidance and not os.path.isfile(args.pkl_path):
        print(f"[WARNING] IRL weights file not found: {args.pkl_path}")
        print("  Options:")
        print("    1. Train IRL first:  python -m MaxEntIRL.run_irl_context_v2")
        print("    2. Run without guidance:  python run_visualize.py --no_guidance")
        print()
        resp = input("Continue without IRL guidance? [y/N]: ").strip().lower()
        if resp != "y":
            sys.exit(1)
        args.no_guidance = True

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Patch config before importing (to avoid default __post_init__ print)
    from MaxEntIRL.irl_config import FeatureExtractionConfig
    config = FeatureExtractionConfig(
        policy_ckpt_dir=args.ckpt_dir,
        policy_ckpt_key=args.ckpt_key,
        scene_location=args.location,
        num_scenes_to_evaluate=args.num_scenes,
        num_scenes_per_batch=args.num_scenes_per_batch,
        guidance_weight=args.guidance_weight,
        horizon=args.horizon,
        seed=args.seed,
    )

    # Override pkl path in inference
    from DiffusionGan.inference import AdversarialIRLDiffusionInference
    inferencer = AdversarialIRLDiffusionInference(config, args.output_dir)

    render_cfg = {
        "size": args.render_size,
        "px_per_m": args.px_per_m,
        "save_every_n_frames": args.save_every_n_frames,
        "draw_mode": args.draw_mode,
    }

    if args.no_guidance:
        # Run without IRL guidance (baseline)
        print("\n=== Running baseline rollout (no IRL guidance) ===\n")
        result_stats = inferencer.run_and_save_results(
            render_to_video=args.render_video,
            render_to_img=args.render_img,
            render_cfg=render_cfg,
        )
    else:
        # Override pkl path
        inferencer.pkl_dir = args.pkl_path
        print(f"\n=== Running with IRL guidance ===")
        print(f"  IRL weights: {args.pkl_path}")
        print(f"  Guidance weight: {args.guidance_weight}\n")
        result_stats = inferencer.inference_adversarial(
            render_to_video=args.render_video,
            render_to_img=args.render_img,
            render_cfg=render_cfg,
        )

    # Print summary
    if result_stats is not None:
        print("\n=== Results Summary ===")
        for k, v in result_stats.items():
            if k == "scene_index":
                continue
            try:
                import numpy as np
                print(f"  {k}: mean={np.nanmean(v):.4f}, std={np.nanstd(v):.4f}")
            except Exception:
                pass

    # Show output locations
    print(f"\n=== Output ===")
    print(f"  Stats:  {args.output_dir}/{args.location}/stats.json")
    if args.render_video or args.render_img:
        print(f"  Videos: {args.output_dir}/{args.location}/viz/")
    print("\nDone!")


if __name__ == "__main__":
    main()
