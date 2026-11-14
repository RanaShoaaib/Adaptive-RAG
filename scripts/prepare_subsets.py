"""Create the fixed train/val/test question subsets (run once, locally).

Examples:
    python scripts/prepare_subsets.py                    # train 1000, val 500, test 500
    python scripts/prepare_subsets.py --val 50 --force   # rebuild only the validation subset

Existing subset files are never overwritten unless --force is given, so a second
run cannot silently change the evaluation questions.
"""
import argparse
import sys

from adaptive_rag.config import (
    SEED,
    TEST_SUBSET_PATH,
    TRAIN_SUBSET_PATH,
    VAL_SUBSET_PATH,
)

SUBSET_PATHS = {
    "train": TRAIN_SUBSET_PATH,
    "val": VAL_SUBSET_PATH,
    "test": TEST_SUBSET_PATH,
}
DEFAULT_SIZES = {"train": 1000, "val": 500, "test": 500}


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create fixed train/val/test question subsets as Parquet files.",
        epilog="With no size flags, all three splits are built at the default sizes "
               f"{DEFAULT_SIZES}. With any size flag, only the named splits are built.",
    )
    for split in SUBSET_PATHS:
        parser.add_argument(f"--{split}", type=_positive_int, metavar="N",
                            help=f"number of {split} questions")
    parser.add_argument("--strategy", choices=("balanced", "stratified"), default="balanced",
                        help="how to allocate questions across sources (default: balanced)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"sampling seed (default: {SEED})")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing subset files")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    sizes = {split: getattr(args, split) for split in SUBSET_PATHS
             if getattr(args, split) is not None}
    if not sizes:
        sizes = dict(DEFAULT_SIZES)

    existing = [SUBSET_PATHS[split] for split in sizes if SUBSET_PATHS[split].exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing subset files (use --force to replace them):",
              file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 1

    # Imported late: loading the dataset library is slow and unnecessary when refusing.
    from adaptive_rag.data import prepare_data

    configs = {split: {"n": n, "strategy": args.strategy} for split, n in sizes.items()}
    prepare_data.prepare_subsets(configs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
