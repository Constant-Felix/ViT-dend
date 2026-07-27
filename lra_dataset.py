"""Official S4 data adapters for the six Long Range Arena tasks.

The raw data lives under one project-level root, while the preprocessing,
tokenization, splitting, and collate behavior are delegated to the local S4
repository. This keeps the benchmark pipeline tied to the exact S4 source used
for the model instead of maintaining a second implementation in this project.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import types
import warnings
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from torch.utils.data import DataLoader, Dataset, Subset

from s4_torchtext_compat import ensure_torchtext_for_s4


TASK_ALIASES = {
    "text": "imdb",
    "image": "cifar",
    "retrieval": "aan",
}

TASK_DATA_DIRS = {
    "imdb": "imdb",
    "cifar": "cifar",
    "listops": "listops-1000",
    "aan": "tsv_data",
    "pathfinder": "pathfinder32",
    "pathx": "pathfinder128",
}

TOKEN_TASKS = {"imdb", "listops", "aan"}


@dataclass(frozen=True)
class LRATaskSpec:
    name: str
    modality: str
    d_input: int
    d_output: int
    sequence_length: int
    data_dir: Path
    vocab_size: Optional[int] = None
    padding_idx: Optional[int] = None


@dataclass
class LRADataBundle:
    task: str
    spec: LRATaskSpec
    loaders: Dict[str, DataLoader]
    datamodule: Any
    validation_uses_test: bool = False


def canonicalize_lra_task(task: str) -> str:
    key = task.lower()
    key = TASK_ALIASES.get(key, key)
    if key not in TASK_DATA_DIRS:
        choices = ", ".join(sorted({*TASK_DATA_DIRS, *TASK_ALIASES}))
        raise KeyError(f"Unknown LRA task '{task}'. Available tasks: {choices}")
    return key


def get_lra_data_dir(root: str | Path, task: str) -> Path:
    task = canonicalize_lra_task(task)
    return Path(root).expanduser().resolve() / TASK_DATA_DIRS[task]


def _validate_data_dir(task: str, data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"LRA data directory does not exist: {data_dir}")

    required: Dict[str, tuple[str, ...]] = {
        "cifar": (
            "cifar-10-batches-py/batches.meta",
            "cifar-10-batches-py/data_batch_1",
            "cifar-10-batches-py/test_batch",
        ),
        "listops": ("basic_train.tsv", "basic_val.tsv", "basic_test.tsv"),
        "aan": (
            "new_aan_pairs.train.tsv",
            "new_aan_pairs.eval.tsv",
            "new_aan_pairs.test.tsv",
        ),
        "pathfinder": (
            "curv_contour_length_14/imgs",
            "curv_contour_length_14/metadata",
        ),
        "pathx": (
            "curv_contour_length_14/imgs",
            "curv_contour_length_14/metadata",
        ),
    }
    missing = [
        name for name in required.get(task, ()) if not (data_dir / name).exists()
    ]
    if missing:
        formatted = ", ".join(str(data_dir / name) for name in missing)
        raise FileNotFoundError(f"Incomplete {task} data; missing: {formatted}")

    if task == "imdb" and not any(data_dir.rglob("imdb-train.arrow")):
        raise FileNotFoundError(
            f"Could not find the Hugging Face IMDB Arrow cache below {data_dir}"
        )


def _prepare_s4_dataloaders_package(s4_root: Path) -> None:
    """Create the package namespace without importing every optional loader."""

    src_module = importlib.import_module("src")
    package_name = "src.dataloaders"
    if package_name in sys.modules:
        return

    package_dir = s4_root / "src" / "dataloaders"
    package = types.ModuleType(package_name)
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(package_dir)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        package_name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[package_name] = package
    setattr(src_module, "dataloaders", package)


def _load_official_s4_dataset_classes(s4_root: str | Path, data_root: Path):
    """Import S4 dataloaders after fixing the import-time DATA_PATH."""

    s4_root = Path(s4_root).expanduser().resolve()
    if not (s4_root / "src" / "dataloaders" / "lra.py").is_file():
        raise FileNotFoundError(
            f"Could not find the official S4 repository at {s4_root}"
        )

    os.environ["DATA_PATH"] = str(data_root)
    if str(s4_root) not in sys.path:
        sys.path.insert(0, str(s4_root))

    torchtext_error = ensure_torchtext_for_s4()
    if torchtext_error is not None:
        warnings.warn(
            "Installed the pure-Python S4 LRA torchtext compatibility layer "
            f"because the system torchtext could not load: {torchtext_error}",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        _prepare_s4_dataloaders_package(s4_root)
        s4_base = importlib.import_module("src.dataloaders.base")
        s4_basic = importlib.import_module("src.dataloaders.basic")
        s4_lra = importlib.import_module("src.dataloaders.lra")
    except Exception as exc:
        raise ImportError(
            "Failed to import the official S4 dataloaders. Run with the Python "
            "environment used by the S4 repository, including compatible "
            "datasets, torchvision, and torchaudio packages."
        ) from exc

    CIFAR10 = s4_basic.CIFAR10
    AAN = s4_lra.AAN
    IMDB = s4_lra.IMDB
    ListOps = s4_lra.ListOps
    PathFinder = s4_lra.PathFinder

    configured_root = Path(s4_base.default_data_path).resolve()
    if configured_root != data_root:
        raise RuntimeError(
            "S4 dataloaders were imported before DATA_PATH was configured: "
            f"expected {data_root}, found {configured_root}. "
            "Start a fresh Python process."
        )

    return {
        "imdb": IMDB,
        "cifar": CIFAR10,
        "listops": ListOps,
        "aan": AAN,
        "pathfinder": PathFinder,
        "pathx": PathFinder,
    }


def _setup_official_datamodule(datamodule: Any, task: str) -> None:
    """Run S4 setup with compatibility for newer Hugging Face datasets."""

    if task != "aan":
        datamodule.setup()
        return

    try:
        from datasets.arrow_dataset import Column
    except ImportError:
        datamodule.setup()
        return

    if getattr(Column, "__add__", None) is not None:
        datamodule.setup()
        return

    # S4 expects Dataset columns to be Python lists and concatenates the two
    # AAN document columns with ``+``. New datasets versions return Column
    # objects instead. Chaining them preserves the exact S4 iteration order.
    Column.__add__ = lambda self, other: chain(self, other)
    try:
        datamodule.setup()
    finally:
        del Column.__add__


def _official_dataset_config(task: str, max_len: Optional[int]) -> Dict[str, Any]:
    # Values are the resolved official S4 dataset and LRA experiment configs.
    # The actual transforms/tokenizers remain implemented by the imported S4 classes.
    configs: Dict[str, Dict[str, Any]] = {
        "imdb": {
            "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,
        },
        "cifar": {
            "permute": None,
            "grayscale": True,
            "tokenize": False,
            "augment": False,
            "cutout": False,
            "random_erasing": False,
            "val_split": 0.1,
            "seed": 42,
        },
        "listops": {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,
        },
        "aan": {
            "l_max": 4000,
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,
        },
        "pathfinder": {
            "resolution": 32,
            "sequential": True,
            "tokenize": False,
            "center": False,
            "pool": 1,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,
        },
        "pathx": {
            "resolution": 128,
            "sequential": True,
            "tokenize": False,
            "center": False,
            "pool": 1,
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,
        },
    }
    config = dict(configs[task])
    if max_len is not None:
        if task not in TOKEN_TASKS:
            raise ValueError("--max-len is only supported for token LRA tasks")
        config["l_max"] = max_len
    return config


def _limited(dataset: Optional[Dataset], limit: Optional[int]) -> Optional[Dataset]:
    if dataset is None or limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def _unwrap_loader(loader, split: str) -> DataLoader:
    if isinstance(loader, Mapping):
        values = list(loader.values())
        if len(values) != 1:
            raise ValueError(f"Expected one {split} resolution, found {len(values)}")
        loader = values[0]
    elif isinstance(loader, (list, tuple)):
        if len(loader) != 1:
            raise ValueError(f"Expected one {split} loader, found {len(loader)}")
        loader = loader[0]
    if not isinstance(loader, DataLoader):
        raise TypeError(
            f"Official S4 returned an invalid {split} loader: {type(loader)}"
        )
    return loader


def _build_task_spec(task: str, data_dir: Path, datamodule: Any) -> LRATaskSpec:
    if task in TOKEN_TASKS:
        vocab_size = int(datamodule.n_tokens)
        padding_idx = int(datamodule.vocab["<pad>"])
        d_input = 1
        modality = "paired_tokens" if task == "aan" else "tokens"
        sequence_length = int(datamodule.l_max)
    else:
        vocab_size = None
        padding_idx = None
        d_input = int(datamodule.d_input)
        modality = "pixels"
        sequence_length = (
            int(datamodule.resolution**2 // datamodule.pool**2)
            if task in {"pathfinder", "pathx"}
            else 1024
        )

    return LRATaskSpec(
        name=task,
        modality=modality,
        d_input=d_input,
        d_output=int(datamodule.d_output),
        sequence_length=sequence_length,
        data_dir=data_dir,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
    )


def get_s4_lra_data(
    task: str,
    root: str | Path,
    s4_root: str | Path,
    batch_size: int,
    num_workers: int = 4,
    max_samples: Optional[Mapping[str, int]] = None,
    max_len: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> LRADataBundle:
    """Build official-S4 datasets and loaders for one canonical LRA task."""

    task = canonicalize_lra_task(task)
    data_root = Path(root).expanduser().resolve()
    data_dir = get_lra_data_dir(data_root, task)
    _validate_data_dir(task, data_dir)

    if task == "imdb":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    classes = _load_official_s4_dataset_classes(s4_root, data_root)
    config = _official_dataset_config(task, max_len=max_len)
    s4_data_dir = data_root if task == "imdb" else data_dir
    datamodule = classes[task](
        _name_="pathfinder" if task == "pathx" else task,
        data_dir=s4_data_dir,
        **config,
    )
    _setup_official_datamodule(datamodule, task)

    max_samples = max_samples or {}
    train_dataset = _limited(datamodule.dataset_train, max_samples.get("train"))
    val_limit = max_samples.get("val", max_samples.get("dev"))
    validation_uses_test = datamodule.dataset_val is None
    val_source = (
        datamodule.dataset_test
        if validation_uses_test
        else datamodule.dataset_val
    )
    val_dataset = _limited(val_source, val_limit)
    test_dataset = _limited(datamodule.dataset_test, max_samples.get("test"))

    if pin_memory is None:
        pin_memory = True
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,  # S4 configs/loader/default.yaml, including eval.
    }
    train_loader = _unwrap_loader(
        datamodule._train_dataloader(train_dataset, **loader_args), "train"
    )
    val_loader = _unwrap_loader(
        datamodule._eval_dataloader(val_dataset, **loader_args), "validation"
    )
    test_loader = _unwrap_loader(
        datamodule._eval_dataloader(test_dataset, **loader_args), "test"
    )

    spec = _build_task_spec(task, data_dir, datamodule)
    return LRADataBundle(
        task=task,
        spec=spec,
        loaders={"train": train_loader, "dev": val_loader, "test": test_loader},
        datamodule=datamodule,
        validation_uses_test=validation_uses_test,
    )


def describe_lra_release(root: str | Path) -> Dict[str, Dict[str, object]]:
    root = Path(root).expanduser().resolve()
    summary: Dict[str, Dict[str, object]] = {}
    for task in TASK_DATA_DIRS:
        data_dir = get_lra_data_dir(root, task)
        try:
            _validate_data_dir(task, data_dir)
            valid = True
            error = None
        except (FileNotFoundError, ValueError) as exc:
            valid = False
            error = str(exc)
        summary[task] = {"path": str(data_dir), "valid": valid, "error": error}
    return summary


if __name__ == "__main__":
    default_root = os.getenv("DATA_PATH", "/data/hyx/ViT-dend/data/lra_release")
    s4_root = Path('/data/hyx/s4').expanduser()
    if s4_root.is_dir() and str(s4_root) not in sys.path:
        sys.path.insert(0, str(s4_root))

    for task_name, info in describe_lra_release(default_root).items():
        print(task_name, info)

    a = get_s4_lra_data(task='aan',root=default_root,s4_root=s4_root,batch_size=1,num_workers=8,pin_memory=True)
    print(len(a.loaders['train']), len(a.loaders['dev']), len(a.loaders['test']))