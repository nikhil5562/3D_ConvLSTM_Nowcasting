import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nowcasting.training.train import EPOCHS
from nowcasting.training.train import train


def main():
    return 0 if train(epochs=EPOCHS, run_tag="") else 1


if __name__ == "__main__":
    raise SystemExit(main())
