from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_CENTER_REGEX
from .inspection import inspect_image
from .search import SearchConfig, run_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam3-rotation-center-stability",
        description=(
            "Rank tomography reconstructions by the number of particles "
            "detected by SAM3."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search", help="Evaluate a center sweep.")
    search.add_argument("--dataset-dir", required=True, type=Path)
    search.add_argument("--checkpoint", required=True, type=Path)
    search.add_argument("--output-dir", required=True, type=Path)
    search.add_argument("--prompt", default="particle")
    search.add_argument("--confidence", type=float, default=0.2)
    search.add_argument("--high-score", type=float, default=0.7)
    search.add_argument("--nms-iou", type=float, default=0.5)
    search.add_argument("--min-area", type=int, default=0)
    search.add_argument("--max-area", type=int)
    search.add_argument("--rolling-radius", type=int, default=5)
    search.add_argument("--glob", default="*.png")
    search.add_argument("--center-regex", default=DEFAULT_CENTER_REGEX)
    search.add_argument("--step", type=int, default=1)
    search.add_argument("--device", default="cuda")
    search.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6]
    )

    inspect = subparsers.add_parser(
        "inspect", help="Inspect mask sizes and duplicate overlap for one image."
    )
    inspect.add_argument("--image", required=True, type=Path)
    inspect.add_argument("--checkpoint", required=True, type=Path)
    inspect.add_argument("--output-dir", required=True, type=Path)
    inspect.add_argument("--prompt", default="particle")
    inspect.add_argument("--confidence", type=float, default=0.2)
    inspect.add_argument("--high-score", type=float, default=0.7)
    inspect.add_argument("--nms-iou", type=float, default=0.5)
    inspect.add_argument("--min-area", type=int, default=0)
    inspect.add_argument("--max-area", type=int)
    inspect.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "search":
        config = SearchConfig(
            dataset_dir=args.dataset_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            prompt=args.prompt,
            confidence_threshold=args.confidence,
            high_score_threshold=args.high_score,
            nms_iou=args.nms_iou,
            min_area=args.min_area,
            max_area=args.max_area,
            rolling_radius=args.rolling_radius,
            glob_pattern=args.glob,
            center_regex=args.center_regex,
            step=args.step,
            device=args.device,
            thresholds=tuple(args.thresholds),
        )
        summary = run_search(config)
        print(json.dumps(summary, indent=2))
    elif args.command == "inspect":
        report = inspect_image(
            args.image,
            args.checkpoint,
            args.output_dir,
            prompt=args.prompt,
            confidence_threshold=args.confidence,
            nms_iou=args.nms_iou,
            high_score_threshold=args.high_score,
            min_area=args.min_area,
            max_area=args.max_area,
            device=args.device,
        )
        print(json.dumps(report, indent=2))
