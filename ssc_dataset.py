from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_ROOT = "/data/hyx/ViT-dend/data/shd/data_shd"
DEFAULT_H5_ROOT = "data"


class SpikingJellyFrameDataset(Dataset):
    """Load pre-integrated SpikingJelly SSC frames from class folders."""

    def __init__(
        self,
        root_path=DEFAULT_ROOT,
        split="train",
        frames_number=250,
        nb_units=700,
        split_by="number",
        frames_dir_name=None,
        dtype=np.float32,
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.frames_number = int(frames_number)
        self.nb_units = int(nb_units)
        self.split_by = split_by
        self.dtype = np.dtype(dtype)

        if frames_dir_name is None:
            frames_dir_name = "frames_fixed_1s_binary_250"

        self.frames_root = self._resolve_frames_root(frames_dir_name)
        self.split_root = self.frames_root / self.split
        self.samples = self._collect_samples(self.split_root)
        if not self.samples:
            raise RuntimeError(f"No .npz frame files found under {self.split_root}")

    def _resolve_frames_root(self, frames_dir_name):
        candidates = [
            self.root_path,
            self.root_path / frames_dir_name,
            self.root_path / "extract" / frames_dir_name,
        ]
        for path in candidates:
            if (path / self.split).exists():
                return path

        raise FileNotFoundError(
            "Could not find SpikingJelly frame directory. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    @staticmethod
    def _sort_key(path):
        try:
            return int(path.stem)
        except ValueError:
            return path.name

    def _collect_samples(self, split_root):
        samples = []
        class_dirs = sorted(
            [path for path in split_root.iterdir() if path.is_dir()],
            key=lambda path: int(path.name) if path.name.isdigit() else path.name,
        )
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for frame_path in sorted(class_dir.glob("*.npz"), key=self._sort_key):
                samples.append((frame_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_path, label = self.samples[index]
        frames = np.load(frame_path, allow_pickle=True)["frames"]
        expected_shape = (self.frames_number, self.nb_units)
        if frames.shape != expected_shape:
            raise ValueError(f"{frame_path} has shape {frames.shape}, expected {expected_shape}")
        frames = np.array(frames.astype(self.dtype, copy=False), copy=True)
        return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)


def create_spikingjelly_frame_dataloader(
    split,
    batch_size,
    root_path=DEFAULT_ROOT,
    frames_number=250,
    nb_units=700,
    split_by="number",
    frames_dir_name=None,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    dtype=np.float32,
):
    dataset = SpikingJellyFrameDataset(
        root_path=root_path,
        split=split,
        frames_number=frames_number,
        nb_units=nb_units,
        split_by=split_by,
        frames_dir_name=frames_dir_name,
        dtype=dtype,
    )
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for the fixed-time SSC/SHD H5 loader. "
            "Install h5py or use create_spikingjelly_frame_dataloader instead."
        ) from exc
    return h5py


class FixedTimeBinH5Dataset(Dataset):
    """Load raw spike-event H5 files and bin events on a fixed real-time axis.

    Each sample is converted from event lists ``times`` and ``units`` into a
    dense tensor with shape ``[nb_steps, nb_units]``.  The time range
    ``[0, max_time]`` is split into exactly ``nb_steps`` equal-width bins.
    """

    def __init__(
        self,
        root_path=DEFAULT_H5_ROOT,
        split="train",
        dataset="SSC",
        nb_steps=250,
        nb_units=700,
        max_time=1.4,
        dtype=np.float32,
        binary=False,
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.dataset = dataset.lower()
        self.nb_steps = int(nb_steps)
        self.nb_units = int(nb_units)
        self.max_time = float(max_time)
        self.dtype = np.dtype(dtype)
        self.binary = bool(binary)

        if self.nb_steps <= 0:
            raise ValueError("nb_steps must be positive")
        if self.nb_units <= 0:
            raise ValueError("nb_units must be positive")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")

        self.h5_path = self._resolve_h5_file()
        self._file = None
        self._times = None
        self._units = None
        self.labels_ = self._read_labels()
        self.num_samples = len(self.labels_)

    def _resolve_h5_file(self):
        file_name = f"{self.dataset}_{self.split}.h5"
        if self.root_path.suffix == ".h5":
            candidates = [self.root_path]
        else:
            candidates = [
                self.root_path / file_name,
                self.root_path / "extract" / file_name,
                self.root_path / self.dataset / "extract" / file_name,
            ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Could not find fixed-time H5 file. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _read_labels(self):
        h5py = _import_h5py()
        with h5py.File(self.h5_path, "r") as h5_file:
            if "labels" not in h5_file:
                raise KeyError(f"{self.h5_path} does not contain a 'labels' dataset")
            return np.asarray(h5_file["labels"], dtype=np.int64)

    def _ensure_open(self):
        if self._file is not None:
            return
        h5py = _import_h5py()
        self._file = h5py.File(self.h5_path, "r")
        if "spikes" not in self._file:
            raise KeyError(f"{self.h5_path} does not contain a 'spikes' group")
        spikes = self._file["spikes"]
        if "times" not in spikes or "units" not in spikes:
            raise KeyError(f"{self.h5_path} spikes group must contain 'times' and 'units'")
        self._times = spikes["times"]
        self._units = spikes["units"]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._times = None
            self._units = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_times"] = None
        state["_units"] = None
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return self.num_samples

    def _events_to_frames(self, times, units):
        frames = np.zeros((self.nb_steps, self.nb_units), dtype=self.dtype)
        if len(times) == 0:
            return frames

        times = np.asarray(times, dtype=np.float64)
        units = np.asarray(units, dtype=np.int64)
        valid = (
            (times >= 0.0)
            & (times <= self.max_time)
            & (units >= 0)
            & (units < self.nb_units)
        )
        if not np.any(valid):
            return frames

        times = times[valid]
        units = units[valid]
        time_index = np.floor(times * self.nb_steps / self.max_time).astype(np.int64)
        time_index = np.clip(time_index, 0, self.nb_steps - 1)

        if self.binary:
            frames[time_index, units] = 1.0
        else:
            np.add.at(frames, (time_index, units), 1.0)
        return frames

    def __getitem__(self, index):
        self._ensure_open()
        frames = self._events_to_frames(self._times[index], self._units[index])
        label = int(self.labels_[index])
        return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)


def create_fixed_time_h5_dataloader(
    split,
    batch_size,
    root_path=DEFAULT_H5_ROOT,
    dataset="SSC",
    nb_steps=250,
    nb_units=700,
    max_time=1.4,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    dtype=np.float32,
    binary=False,
):
    dataset_obj = FixedTimeBinH5Dataset(
        root_path=root_path,
        split=split,
        dataset=dataset,
        nb_steps=nb_steps,
        nb_units=nb_units,
        max_time=max_time,
        dtype=dtype,
        binary=binary,
    )
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset_obj, **kwargs)


def asrc_style_events_to_frames(
    times,
    units,
    nb_steps=250,
    nb_units=700,
    max_time=1.4,
    n_bins=5,
    dtype=np.float32,
    time_bins=None,
):
    """Replicate the ASRC-SNN SSC event-to-frame preprocessing.

    The ASRC-SNN code first maps event times with ``np.digitize`` over
    ``np.linspace(0, max_time, nb_steps)``, writes binary spikes into 700 input
    channels, then sums every ``n_bins`` adjacent channels.
    """
    nb_steps = int(nb_steps)
    nb_units = int(nb_units)
    n_bins = int(n_bins)
    if nb_steps <= 0:
        raise ValueError("nb_steps must be positive")
    if nb_units <= 0:
        raise ValueError("nb_units must be positive")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if nb_units % n_bins != 0:
        raise ValueError("nb_units must be divisible by n_bins")

    dtype = np.dtype(dtype)
    dense = np.zeros((nb_steps, nb_units), dtype=dtype)
    if len(times) > 0:
        if time_bins is None:
            time_bins = np.linspace(0, max_time, num=nb_steps)
        time_index = np.digitize(times, time_bins)
        units = np.asarray(units, dtype=np.int64)
        valid = (
            (time_index >= 0)
            & (time_index < nb_steps)
            & (units >= 0)
            & (units < nb_units)
        )
        if np.any(valid):
            dense[time_index[valid], units[valid]] = 1.0

    if n_bins > 1:
        binned_len = nb_units // n_bins
        dense = dense.reshape(nb_steps, binned_len, n_bins).sum(axis=-1)
        dense = dense.astype(dtype, copy=False)
    return dense


class ASRCStyleSSCH5Dataset(FixedTimeBinH5Dataset):
    """Load SSC H5 files with the preprocessing used by the ASRC-SNN code."""

    def __init__(
        self,
        root_path=DEFAULT_H5_ROOT,
        split="train",
        dataset="SSC",
        nb_steps=250,
        nb_units=700,
        max_time=1.4,
        n_bins=5,
        dtype=np.float32,
    ):
        self.n_bins = int(n_bins)
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if int(nb_units) % self.n_bins != 0:
            raise ValueError("nb_units must be divisible by n_bins")
        self.time_bins = np.linspace(0, float(max_time), num=int(nb_steps))
        super().__init__(
            root_path=root_path,
            split=split,
            dataset=dataset,
            nb_steps=nb_steps,
            nb_units=nb_units,
            max_time=max_time,
            dtype=dtype,
            binary=True,
        )
        self.input_dim = self.nb_units // self.n_bins

    def _events_to_frames(self, times, units):
        return asrc_style_events_to_frames(
            times,
            units,
            nb_steps=self.nb_steps,
            nb_units=self.nb_units,
            max_time=self.max_time,
            n_bins=self.n_bins,
            dtype=self.dtype,
            time_bins=self.time_bins,
        )
    
def create_asrc_style_ssc_dataloader(
    split,
    batch_size,
    root_path=DEFAULT_H5_ROOT,
    dataset="SSC",
    nb_steps=250,
    nb_units=700,
    max_time=1.4,
    n_bins=5,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
    dtype=np.float32,
):
    dataset_obj = ASRCStyleSSCH5Dataset(
        root_path=root_path,
        split=split,
        dataset=dataset,
        nb_steps=nb_steps,
        nb_units=nb_units,
        max_time=max_time,
        n_bins=n_bins,
        dtype=dtype,
    )
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset_obj, **kwargs)    