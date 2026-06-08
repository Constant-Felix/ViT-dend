"""Data loading utilities for local LRA pickle-zip datasets.

Expected directory layout:

    lra_release/
      IMDB/lra-text.{train,dev,test}.pickle.zip
      cifar/lra-image.{train,dev,test}.pickle.zip
      listops/lra-listops.{train,dev,test}.pickle.zip
      pathfinder/lra-pathfinder32-curv_contour_length_14.{train,dev,test}.pickle.zip

Each pickle file is a list of dicts with ``input_ids_0`` and ``label``.
Token tasks are trimmed back to their non-padding length and collated with
dynamic padding plus ``lengths``, matching the official S4 loaders. Pixel tasks
apply the official S4 transforms for the corresponding LRA task.
"""

from __future__ import annotations

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LRATaskSpec:
    name: str
    dirname: str
    filename_prefix: str
    modality: str
    d_input: int
    d_output: int
    sequence_length: int
    vocab_size: Optional[int] = None
    padding_idx: Optional[int] = None
    pixel_normalize: bool = False
    pixel_mean: Optional[float] = None
    pixel_std: Optional[float] = None


LRA_TASKS: Dict[str, LRATaskSpec] = {
    "imdb": LRATaskSpec(
        name="imdb",
        dirname="IMDB",
        filename_prefix="lra-text",
        modality="tokens",
        d_input=1,
        d_output=2,
        sequence_length=4096,
        vocab_size=241,
        padding_idx=0,
    ),
    "text": LRATaskSpec(
        name="imdb",
        dirname="IMDB",
        filename_prefix="lra-text",
        modality="tokens",
        d_input=1,
        d_output=2,
        sequence_length=4096,
        vocab_size=241,
        padding_idx=0,
    ),
    "cifar": LRATaskSpec(
        name="cifar",
        dirname="cifar",
        filename_prefix="lra-image",
        modality="pixels",
        d_input=1,
        d_output=10,
        sequence_length=1024,
        vocab_size=None,
        pixel_normalize=True,
        pixel_mean=122.6 / 255.0,
        pixel_std=61.0 / 255.0,
    ),
    "image": LRATaskSpec(
        name="cifar",
        dirname="cifar",
        filename_prefix="lra-image",
        modality="pixels",
        d_input=1,
        d_output=10,
        sequence_length=1024,
        vocab_size=None,
        pixel_normalize=True,
        pixel_mean=122.6 / 255.0,
        pixel_std=61.0 / 255.0,
    ),
    "listops": LRATaskSpec(
        name="listops",
        dirname="listops",
        filename_prefix="lra-listops",
        modality="tokens",
        d_input=1,
        d_output=10,
        sequence_length=2048,
        vocab_size=15,
        padding_idx=0,
    ),
    "pathfinder": LRATaskSpec(
        name="pathfinder",
        dirname="pathfinder",
        filename_prefix="lra-pathfinder32-curv_contour_length_14",
        modality="pixels",
        d_input=1,
        d_output=2,
        sequence_length=1024,
        vocab_size=None,
        pixel_normalize=True,
    ),
}


SPLIT_ALIASES = {
    "train": "train",
    "val": "dev",
    "valid": "dev",
    "validation": "dev",
    "dev": "dev",
    "test": "test",
}


def get_lra_task_spec(task: str) -> LRATaskSpec:
    key = task.lower()
    if key not in LRA_TASKS:
        choices = ", ".join(sorted(LRA_TASKS))
        raise KeyError(f"Unknown LRA task '{task}'. Available tasks: {choices}")
    return LRA_TASKS[key]


def normalize_split(split: str) -> str:
    key = split.lower()
    if key not in SPLIT_ALIASES:
        choices = ", ".join(sorted(SPLIT_ALIASES))
        raise KeyError(f"Unknown split '{split}'. Available splits: {choices}")
    return SPLIT_ALIASES[key]


def get_lra_zip_path(root: str | Path, task: str, split: str) -> Path:
    spec = get_lra_task_spec(task)
    split = normalize_split(split)
    path = Path(root) / spec.dirname / f"{spec.filename_prefix}.{split}.pickle.zip"
    if not path.exists():
        raise FileNotFoundError(f"Could not find LRA split file: {path}")
    return path


class LRAPickleZipDataset(Dataset):
    """Dataset for Kaggle-style LRA ``.pickle.zip`` split files."""

    def __init__(
        self,
        root: str | Path,
        task: str,
        split: str,
        max_samples: Optional[int] = None,
        max_len: Optional[int] = None,
        pixel_normalize: Optional[bool] = None,
        return_lengths: Optional[bool] = None,
        trim_padding: Optional[bool] = None,
        preload: bool = True,
    ) -> None:
        self.spec = get_lra_task_spec(task)
        self.split = normalize_split(split)
        self.path = get_lra_zip_path(root, task, split)
        self.max_samples = max_samples
        self.max_len = max_len
        self.pixel_normalize = self.spec.pixel_normalize if pixel_normalize is None else pixel_normalize
        self.return_lengths = self.spec.modality == "tokens" if return_lengths is None else return_lengths
        self.trim_padding = self.spec.modality == "tokens" if trim_padding is None else trim_padding

        if not preload:
            raise ValueError("LRAPickleZipDataset currently requires preload=True")
        self.data = self._load_pickle_zip(self.path)
        if max_samples is not None:
            self.data = self.data[:max_samples]

    @staticmethod
    def _load_pickle_zip(path: Path):
        with zipfile.ZipFile(path) as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            if len(names) != 1:
                raise ValueError(f"Expected one pickle file inside {path}, found {names}")
            with zf.open(names[0]) as f:
                return pickle.load(f)

    def _process_pixel_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.pixel_normalize:
            x = x / 255.0
        if self.spec.pixel_mean is not None or self.spec.pixel_std is not None:
            if not self.pixel_normalize:
                raise ValueError("pixel_mean/pixel_std expect pixel_normalize=True")
            mean = 0.0 if self.spec.pixel_mean is None else self.spec.pixel_mean
            std = 1.0 if self.spec.pixel_std is None else self.spec.pixel_std
            x = (x - mean) / std
        return x.unsqueeze(-1)

    def _process_token_input(self, x: torch.Tensor):
        x = x.long()
        length = x.numel()
        if self.trim_padding and self.spec.padding_idx is not None:
            non_padding = torch.nonzero(x != self.spec.padding_idx, as_tuple=False)
            length = int(non_padding[-1, 0].item()) + 1 if non_padding.numel() else 0
            x = x[:length]
        return x, torch.as_tensor(length, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        x = torch.as_tensor(item["input_ids_0"])
        y = torch.as_tensor(item["label"], dtype=torch.long)

        if self.max_len is not None:
            x = x[: self.max_len]

        if self.spec.modality == "pixels":
            x = self._process_pixel_input(x)
            return x, y

        x, length = self._process_token_input(x)
        if self.return_lengths:
            return x, y, length
        return x, y


def collate_lra_batch(batch, padding_idx: int = 0):
    if len(batch[0]) == 3:
        xs, ys, lengths = zip(*batch)
        lengths = torch.stack(lengths).long()
    else:
        xs, ys = zip(*batch)
        lengths = None
    first = xs[0]

    if first.dim() == 1:
        if all(item.size(0) == first.size(0) for item in xs):
            x = torch.stack(xs, dim=0)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                xs,
                batch_first=True,
                padding_value=padding_idx,
            )
        if lengths is None:
            lengths = torch.as_tensor([item.size(0) for item in xs], dtype=torch.long)
    elif first.dim() == 2:
        if all(item.size(0) == first.size(0) for item in xs):
            x = torch.stack(xs, dim=0)
        else:
            x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    else:
        raise ValueError(f"Unsupported LRA sample rank: {first.dim()}")

    y = torch.stack(ys, dim=0).long()
    if lengths is not None:
        return x, y, {"lengths": lengths}
    return x, y


def get_lra_datasets(
    task: str,
    root: str | Path = "lra_release",
    max_samples: Optional[Mapping[str, int]] = None,
    max_len: Optional[int] = None,
    pixel_normalize: Optional[bool] = None,
    return_lengths: Optional[bool] = None,
    trim_padding: Optional[bool] = None,
) -> Dict[str, LRAPickleZipDataset]:
    max_samples = max_samples or {}
    return {
        split: LRAPickleZipDataset(
            root=root,
            task=task,
            split=split,
            max_samples=max_samples.get(split),
            max_len=max_len,
            pixel_normalize=pixel_normalize,
            return_lengths=return_lengths,
            trim_padding=trim_padding,
        )
        for split in ("train", "dev", "test")
    }


def get_lra_dataloaders(
    task: str,
    root: str | Path = "lra_release",
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: Optional[Mapping[str, int]] = None,
    max_len: Optional[int] = None,
    pixel_normalize: Optional[bool] = None,
    return_lengths: Optional[bool] = None,
    trim_padding: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    datasets = get_lra_datasets(
        task=task,
        root=root,
        max_samples=max_samples,
        max_len=max_len,
        pixel_normalize=pixel_normalize,
        return_lengths=return_lengths,
        trim_padding=trim_padding,
    )
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batch, padding_idx=dataset.spec.padding_idx or 0: collate_lra_batch(
                batch,
                padding_idx=padding_idx,
            ),
        )
        for split, dataset in datasets.items()
    }


def describe_lra_release(root: str | Path = "lra_release") -> Dict[str, Dict[str, object]]:
    summary = {}
    for task_key in ("imdb", "cifar", "listops", "pathfinder"):
        spec = get_lra_task_spec(task_key)
        split_info = {}
        for split in ("train", "dev", "test"):
            path = get_lra_zip_path(root, task_key, split)
            with zipfile.ZipFile(path) as zf:
                name = zf.namelist()[0]
                size = zf.getinfo(name).file_size
            split_info[split] = {"path": str(path), "uncompressed_bytes": size}
        summary[task_key] = {
            "modality": spec.modality,
            "d_input": spec.d_input,
            "d_output": spec.d_output,
            "sequence_length": spec.sequence_length,
            "vocab_size": spec.vocab_size,
            "padding_idx": spec.padding_idx,
            "pixel_normalize": spec.pixel_normalize,
            "pixel_mean": spec.pixel_mean,
            "pixel_std": spec.pixel_std,
            "splits": split_info,
        }
    return summary


if __name__ == "__main__":
    for task, info in describe_lra_release().items():
        print(task, info)