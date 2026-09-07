import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.config import INPUT_LENGTH
from nowcasting.config import OUTPUT_LENGTH
from nowcasting.data.cartesian_grid import create_gridded_cubes
from nowcasting.data.l2b_reader import diagnose_sequence_capacity
from nowcasting.data.l2b_reader import discover_l2b_candidates
from nowcasting.data.provenance import ProvenanceMismatchError
from nowcasting.data.tensorize import tensorize_cube
from nowcasting.paths import GRIDDED_DATA_DIR
from nowcasting.paths import L2B_INPUT_DIR
from nowcasting.paths import PROCESSED_DATA_DIR


def build_parser():
    parser = argparse.ArgumentParser(description="Run L2B -> gridded cube -> ConvLSTM tensor pipeline.")
    parser.add_argument("--input", type=Path, default=L2B_INPUT_DIR, help="Folder with RCTLS_*_L2B_STD.nc files.")
    parser.add_argument("--cube-output", type=Path, default=GRIDDED_DATA_DIR, help="Folder for *_gridded.nc cubes.")
    parser.add_argument("--tensor-output", type=Path, default=PROCESSED_DATA_DIR, help="Folder for compact .npy tensors.")
    parser.add_argument("--min-size-mb", type=float, default=100.0, help="Fast full-volume size prefilter.")
    parser.add_argument("--min-sweeps", type=int, default=11, help="Minimum sweeps required after loading.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum scans to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument(
        "--allow-insufficient-sequence",
        action="store_true",
        help=(
            "Process scans for pipeline QA even when the archive cannot form one "
            "complete training sequence."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.input.is_dir():
        print(f"ERROR: L2B input directory does not exist: {args.input}")
        print("Pass the actual MOSDAC L2B folder with --input <directory>.")
        return 2

    candidates = discover_l2b_candidates(args.input, min_size_mb=args.min_size_mb)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    required_frames = INPUT_LENGTH + OUTPUT_LENGTH
    capacity = diagnose_sequence_capacity(candidates, frames_per_sequence=required_frames)
    print(
        "Sequence preflight: "
        f"frames={capacity['frame_count']}, runs={capacity['run_lengths']}, "
        f"longest={capacity['longest_run']}, required={required_frames}, "
        f"possible_sequences={capacity['sequence_count']}"
    )
    if not candidates:
        print("ERROR: No probable full-volume RCTLS L2B files were found.")
        return 1
    if capacity["sequence_count"] == 0 and not args.allow_insufficient_sequence:
        print(
            "ERROR: The archive cannot form a complete training sequence. "
            "Acquire more continuous full-volume scans, or pass "
            "--allow-insufficient-sequence only for preprocessing QA."
        )
        return 3

    try:
        cubes = create_gridded_cubes(
            input_dir=args.input,
            output_dir=args.cube_output,
            min_size_mb=args.min_size_mb,
            min_sweeps=args.min_sweeps,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        tensors = [
            tensorize_cube(cube, output_dir=args.tensor_output, overwrite=args.overwrite)
            for cube in cubes
        ]
    except ProvenanceMismatchError as exc:
        print(f"ERROR: {exc}")
        return 4

    print(f"Created or reused {len(cubes)} gridded cubes in {args.cube_output}")
    print(f"Created or reused {len(tensors)} ConvLSTM tensors in {args.tensor_output}")
    return 0 if tensors else 1


if __name__ == "__main__":
    raise SystemExit(main())
