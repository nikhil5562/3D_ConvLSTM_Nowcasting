import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.data.tensorize import tensorize_directory
from nowcasting.paths import GRIDDED_DATA_DIR
from nowcasting.paths import PROCESSED_DATA_DIR


def build_parser():
    parser = argparse.ArgumentParser(description="Create compact ConvLSTM tensors from gridded cubes.")
    parser.add_argument("--input", type=Path, default=GRIDDED_DATA_DIR, help="Folder with *_gridded.nc cubes.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR, help="Folder for compact .npy tensors.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum cubes to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate tensors that already exist.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = tensorize_directory(
        input_dir=args.input,
        output_dir=args.output,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(f"Created or reused {len(outputs)} ConvLSTM tensors in {args.output}")
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
