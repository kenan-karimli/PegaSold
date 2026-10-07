"""Shim: single source of truth lives in src/data_prep.py."""
from src.data_prep import parse_args, run


def main() -> None:
    args = parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
