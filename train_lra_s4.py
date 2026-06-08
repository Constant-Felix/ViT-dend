"""Train S4/MMDEND-style S4 models on local LRA pickle-zip datasets.

This script mirrors the standalone training style of ``/Users/yuxuan/s4/example.py``
while using the local ``lra_dataset.py`` loader and ``model/s4_lra.py`` model
builder. By default it trains the activation-replaced DEND+SOMA S4 model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from tqdm.auto import tqdm

from lra_dataset import get_lra_dataloaders, get_lra_task_spec


SCHEDULER_PRESETS: Dict[str, Dict[str, int]] = {
    "aan": {"num_training_steps": 50000, "num_warmup_steps": 5000},
    "retrieval": {"num_training_steps": 50000, "num_warmup_steps": 5000},
    "cifar": {"num_training_steps": 180000, "num_warmup_steps": 900},
    "image": {"num_training_steps": 180000, "num_warmup_steps": 900},
    "imdb": {"num_training_steps": 50000, "num_warmup_steps": 5000},
    "text": {"num_training_steps": 50000, "num_warmup_steps": 5000},
    "pathfinder": {"num_training_steps": 500000, "num_warmup_steps": 50000},
    "listops": {"num_training_steps": 80000, "num_warmup_steps": 1000},
    "pathx": {"num_training_steps": 500000, "num_warmup_steps": 50000},
}


def load_s4_lra_module():
    path = Path(__file__).resolve().parent / "model" / "s4_lra.py"
    spec = importlib.util.spec_from_file_location("s4_lra_file", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load S4 LRA module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """Create AdamW groups, respecting S4 parameters with custom ``_optim`` attrs."""

    all_parameters = list(model.parameters())
    base_params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(base_params, lr=lr, weight_decay=weight_decay)

    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(items) for items in sorted(dict.fromkeys(frozenset(hp.items()) for hp in hps))]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    keys = sorted({key for hp in hps for key in hp})
    for i, group in enumerate(optimizer.param_groups):
        group_hps = " ".join(f"{key}={group.get(key, None)}" for key in keys)
        print(f"Optimizer group {i}: {len(group['params'])} tensors {group_hps}".rstrip())

    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    num_training_steps: Optional[int],
    num_warmup_steps: Optional[int],
):
    if scheduler_name == "none":
        return None, "epoch"
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs), "epoch"

    total_steps = num_training_steps or epochs * steps_per_epoch
    warmup_steps = num_warmup_steps
    if warmup_steps is None:
        warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, float(step - warmup_steps) / float(total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"


def unpack_batch(batch, device: torch.device):
    if len(batch) == 3:
        inputs, targets, extra = batch
        lengths = extra.get("lengths")
    else:
        inputs, targets = batch
        lengths = None

    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if lengths is not None:
        lengths = lengths.to(device, non_blocking=True)
    return inputs, targets, lengths


def run_epoch(
    loader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler,
    scheduler_interval: str,
    scaler: Optional[amp.GradScaler],
    device: torch.device,
    train: bool,
    print_freq: int,
    epoch: int,
) -> Tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    phase = "train" if train else "eval"

    iterator = tqdm(enumerate(loader), total=len(loader), leave=False)
    for batch_idx, batch in iterator:
        inputs, targets, lengths = unpack_batch(batch, device)

        with torch.set_grad_enabled(train):
            with amp.autocast(enabled=scaler is not None):
                logits = model(inputs, lengths=lengths)
                loss = criterion(logits, targets)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += batch_size

        if print_freq > 0 and (batch_idx % print_freq == 0 or batch_idx + 1 == len(loader)):
            avg_loss = total_loss / max(total_seen, 1)
            avg_acc = 100.0 * total_correct / max(total_seen, 1)
            iterator.set_description(
                f"{phase} epoch={epoch} batch={batch_idx + 1}/{len(loader)} "
                f"loss={avg_loss:.4f} acc={avg_acc:.2f}"
            )

    return total_loss / total_seen, 100.0 * total_correct / total_seen


def save_checkpoint(state: dict, output_dir: Path, is_best: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "checkpoint.pth.tar"
    torch.save(state, latest)
    if is_best:
        torch.save(state, output_dir / "model_best.pth.tar")

# python train_lra_s4.py --task cifar --eval-test-every-epoch --activation dend_soma --dend-branches 8 --dend-compartments 2 --dend-branch-degree 2 --soma-psn-order 1024 --amp --batch-size 1

def parse_args():
    parser = argparse.ArgumentParser(description="Train S4-LRA with optional DEND+SOMA activation.")
    parser.add_argument("--task", default="listops", choices=["listops", "imdb", "text", "cifar", "image", "pathfinder"])
    parser.add_argument("--root", default="/data/hyx/ViT-dend/data/lra_release", help="Root directory containing local LRA pickle zip folders.")
    parser.add_argument("--s4-root", default="/data/hyx/s4", help="Official S4 repo root to add to PYTHONPATH.")
    parser.add_argument("--backend", default="official", choices=["official", "fallback", "auto"])
    parser.add_argument("--activation", default="dend_soma", choices=["dend_soma", "standard"])
    parser.add_argument("--output-dir", default="", help="Directory for args/checkpoints. Default creates exp/lra-*.")
    parser.add_argument("--resume", default="", help="Resume from checkpoint.pth.tar/model_best.pth.tar.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", "--wd", dest="weight_decay", type=float, default=None)
    parser.add_argument("--scheduler", default="cosine-warmup", choices=["cosine-warmup", "cosine", "none"])
    parser.add_argument("--num-training-steps", type=int, default=None)
    parser.add_argument("--num-warmup-steps", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-test-every-epoch", action="store_true")
    parser.add_argument("--zero-pad-embedding", action="store_true", help="Use padding_idx=0 in token embedding.")

    parser.add_argument("--dend-branches", type=int, default=2)
    parser.add_argument("--dend-compartments", type=int, default=4)
    parser.add_argument("--dend-branch-degree", type=int, default=2)
    parser.add_argument("--soma-psn-order", type=int, default=32)
    parser.add_argument("--soma-psn-backend", default="gemm", choices=["gemm", "conv"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s4_root = Path(args.s4_root).expanduser()
    if s4_root.is_dir() and str(s4_root) not in sys.path:
        sys.path.insert(0, str(s4_root))

    s4_lra = load_s4_lra_module()
    task_key = args.task.lower()
    spec = get_lra_task_spec(task_key)
    training_preset = dict(s4_lra.LRA_TRAINING_PRESETS.get(task_key, {}))
    scheduler_preset = dict(SCHEDULER_PRESETS.get(task_key, {}))

    args.epochs = args.epochs or int(training_preset.get("epochs", 1))
    args.batch_size = args.batch_size or int(training_preset.get("batch_size", 32))
    args.lr = args.lr if args.lr is not None else float(training_preset.get("lr", 0.01))
    args.weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(training_preset.get("weight_decay", 0.05))
    )
    if args.num_training_steps is None:
        args.num_training_steps = scheduler_preset.get("num_training_steps")
    if args.num_warmup_steps is None:
        args.num_warmup_steps = scheduler_preset.get("num_warmup_steps")
    if args.seed is None:
        args.seed = 2222 if task_key in {"cifar", "image"} else 3333

    set_seed(args.seed)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    max_samples = {
        "train": args.max_train_samples,
        "dev": args.max_val_samples,
        "test": args.max_test_samples,
    }
    max_samples = {k: v for k, v in max_samples.items() if v is not None}
    loaders = get_lra_dataloaders(
        task_key,
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_samples=max_samples,
        max_len=args.max_len,
    )

    model_overrides = {}
    if args.d_model is not None:
        model_overrides["d_model"] = args.d_model
    if args.d_state is not None:
        model_overrides["d_state"] = args.d_state
    if args.n_layers is not None:
        model_overrides["n_layers"] = args.n_layers
    if args.dropout is not None:
        model_overrides["dropout"] = args.dropout
    if args.activation == "dend_soma":
        model_overrides.update(
            {
                "dend_soma_num_branches": args.dend_branches,
                "dend_soma_compartments_per_branch": args.dend_compartments,
                "dend_soma_branch_degree": args.dend_branch_degree,
                "dend_soma_psn_order": args.soma_psn_order,
                "dend_soma_psn_backend": args.soma_psn_backend,
            }
        )

    builder = (
        s4_lra.build_dend_soma_s4_lra
        if args.activation == "dend_soma"
        else s4_lra.build_standard_s4_lra
    )
    embedding_padding_idx = spec.padding_idx if args.zero_pad_embedding else None
    model = builder(
        task_key,
        d_input=spec.d_input,
        d_output=spec.d_output,
        vocab_size=spec.vocab_size,
        backend=args.backend,
        padding_idx=embedding_padding_idx,
        **model_overrides,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(model, args.lr, args.weight_decay)
    scheduler, scheduler_interval = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        len(loaders["train"]),
        args.num_training_steps,
        args.num_warmup_steps,
    )
    scaler = amp.GradScaler() if args.amp and device.type == "cuda" else None

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path("exp") / f"lra-{task_key}-{args.activation}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val", -1.0))
        print(f"Resumed {args.resume} at epoch {start_epoch} best_val={best_val:.2f}")

    with (output_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Task={task_key} spec={spec}")
    print(f"Device={device} backend={args.backend} activation={args.activation}")
    print(f"Output dir={output_dir}")
    print(
        f"Epochs={args.epochs} batch_size={args.batch_size} "
        f"train_batches={len(loaders['train'])} scheduler={args.scheduler}/{scheduler_interval}"
    )

    for epoch in range(start_epoch, args.epochs):
        tic = time.time()
        train_loss, train_acc = run_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            scheduler,
            scheduler_interval,
            scaler,
            device,
            train=True,
            print_freq=args.print_freq,
            epoch=epoch + 1,
        )
        val_loss, val_acc = run_epoch(
            loaders["dev"],
            model,
            criterion,
            optimizer=None,
            scheduler=None,
            scheduler_interval=scheduler_interval,
            scaler=None,
            device=device,
            train=False,
            print_freq=args.print_freq,
            epoch=epoch + 1,
        )
        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()

        test_loss = test_acc = None
        if args.eval_test_every_epoch:
            test_loss, test_acc = run_epoch(
                loaders["test"],
                model,
                criterion,
                optimizer=None,
                scheduler=None,
                scheduler_interval=scheduler_interval,
                scaler=None,
                device=device,
                train=False,
                print_freq=args.print_freq,
                epoch=epoch + 1,
            )

        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)
        state = {
            "epoch": epoch + 1,
            "best_val": best_val,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "args": vars(args),
            "task_spec": spec.__dict__,
        }
        save_checkpoint(state, output_dir, is_best=is_best)

        lr = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f} "
            f"best_val={best_val:.2f} lr={lr:.6g} time={time.time() - tic:.1f}s"
        )
        if test_acc is not None:
            msg += f" test_loss={test_loss:.4f} test_acc={test_acc:.2f}"
        print(msg, flush=True)

    best_path = output_dir / "model_best.pth.tar"
    if best_path.exists():
        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["state_dict"])
    test_loss, test_acc = run_epoch(
        loaders["test"],
        model,
        criterion,
        optimizer=None,
        scheduler=None,
        scheduler_interval=scheduler_interval,
        scaler=None,
        device=device,
        train=False,
        print_freq=args.print_freq,
        epoch=args.epochs,
    )
    print(f"Best checkpoint test_loss={test_loss:.4f} test_acc={test_acc:.2f}")


if __name__ == "__main__":
    main()