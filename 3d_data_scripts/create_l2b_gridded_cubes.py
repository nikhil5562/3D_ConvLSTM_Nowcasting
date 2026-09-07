import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.data.cartesian_grid import create_gridded_cubes
from nowcasting.paths import GRIDDED_DATA_DIR
from nowcasting.paths import L2B_INPUT_DIR


def build_parser():
    parser = argparse.ArgumentParser(description="Create 81x481x481 gridded cubes from RCTLS L2B files.")
    parser.add_argument("--input", type=Path, default=L2B_INPUT_DIR, help="Folder with RCTLS_*_L2B_STD.nc files.")
    parser.add_argument("--output", type=Path, default=GRIDDED_DATA_DIR, help="Folder for *_gridded.nc cubes.")
    parser.add_argument("--min-size-mb", type=float, default=100.0, help="Fast full-volume size prefilter.")
    parser.add_argument("--min-sweeps", type=int, default=11, help="Minimum sweeps required after loading.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum scans to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cubes that already exist.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = create_gridded_cubes(
        input_dir=args.input,
        output_dir=args.output,
        min_size_mb=args.min_size_mb,
        min_sweeps=args.min_sweeps,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(f"Created or reused {len(outputs)} gridded cubes in {args.output}")
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
