import argparse

from ssc_dataset import DEFAULT_ROOT, create_spikingjelly_frame_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal SSC loader with SpikingJelly frames_number=250 split_by=number."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Path to extract/ or frames_number_250_split_by_number/.",
    )
    parser.add_argument("--frames-number", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "valid", "test"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for split in args.splits:
        loader = create_spikingjelly_frame_dataloader(
            split=split,
            batch_size=args.batch_size,
            root_path=args.root,
            frames_number=args.frames_number,
            split_by="number",
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=True,
        )

        x, y = next(iter(loader))
        print(
            f"{split}: samples={len(loader.dataset)}, "
            f"batch_x_shape={tuple(x.shape)}, batch_x_dtype={x.dtype}, "
            f"batch_y_shape={tuple(y.shape)}, batch_y_dtype={y.dtype}"
        )


if __name__ == "__main__":
    main()