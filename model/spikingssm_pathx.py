"""SpikingSSM PathX backbone with the neuron replaced by DEND+SOMA.

This module mirrors ``SDN/models/spike/ss4d.py`` for the PathX experiment:
each layer uses the official S4 ``SSMKernelDiag`` with ``diag-lin``
initialization, bidirectional FFT convolution, a pointwise Conv1d+GLU output
projection, and an outer pre-norm residual block. The only architectural
substitution is the activation between the S4D convolution and output
projection, where this project uses its channel-preserving DEND+SOMA module.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _load_s4_lra_components():
    """Load local S4 helpers without executing the legacy ``model`` package."""

    module_name = "_vit_dend_s4_lra_shared"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        path = Path(__file__).resolve().with_name("s4_lra.py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local S4 LRA helpers from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.DendSomaS4Activation, module.DropoutNd


DendSomaS4Activation, DropoutNd = _load_s4_lra_components()


S4DKernelFactory = Callable[..., nn.Module]
ActivationFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class SpikingSSMPathXConfig:
    """PathX model values from ``SDN/configs/experiment/spikingssm/pathx.yaml``."""

    d_input: int
    d_output: int
    d_model: int = 256
    d_state: int = 64
    n_layers: int = 6
    dropout: float = 0.0
    prenorm: bool = True
    norm: str = "batch"
    bidirectional: bool = True
    layer_lr: float = 0.001
    dt_min: float = 0.0001
    dt_max: float = 0.1
    sequence_length: int = 16384


def resolve_official_s4d_kernel() -> S4DKernelFactory:
    """Load the same trainable-B S4D kernel used by the SDN repository."""

    try:
        from src.models.sequence.kernels.ssm import SSMKernelDiag
    except ImportError as exc:
        raise ImportError(
            "Could not import official S4 SSMKernelDiag. Add the official S4 "
            "repository root to PYTHONPATH or pass it with --s4-root."
        ) from exc
    return SSMKernelDiag


class SpikingSSMS4DLayer(nn.Module):
    """Source-faithful SDN ``SS4D`` layer with DEND+SOMA as its neuron."""

    def __init__(
        self,
        d_model: int,
        activation_factory: ActivationFactory,
        *,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        bidirectional: bool = True,
        layer_lr: float = 0.001,
        dt_min: float = 0.0001,
        dt_max: float = 0.1,
        kernel_factory: Optional[S4DKernelFactory] = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_state <= 0:
            raise ValueError("d_model and d_state must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.h = d_model
        self.n = d_state
        self.transposed = transposed
        self.bidirectional = bidirectional

        # SDN initializes D before doubling the kernel channels for the two
        # temporal directions. PathX uses one output channel after combining.
        self.D = nn.Parameter(torch.randn(1, d_model))
        kernel_channels = 2 if bidirectional else 1
        if kernel_factory is None:
            kernel_factory = resolve_official_s4d_kernel()
        self.kernel = kernel_factory(
            d_model=d_model,
            d_state=d_state,
            channels=kernel_channels,
            init="diag-lin",
            lr=layer_lr,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        # This is the sole intentional replacement of SDN's threshold scaling
        # and spiking neuron. Everything after it follows SS4D unchanged.
        self.activation = activation_factory(d_model)
        self.dropout = (
            DropoutNd(dropout, tie=True, transposed=True)
            if dropout > 0.0
            else nn.Identity()
        )
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: Tensor, **kwargs):
        """Map ``(B, H, L)`` to ``(B, H, L)`` as in SDN's ``SS4D``."""

        del kwargs
        if not self.transposed:
            u = u.transpose(-1, -2)
        if u.dim() != 3 or u.size(1) != self.h:
            raise ValueError(
                f"Expected input shape (B, {self.h}, L), got {tuple(u.shape)}"
            )

        length = u.size(-1)
        kernel, _ = self.kernel(L=length)
        if self.bidirectional:
            if kernel.size(0) % 2 != 0:
                raise RuntimeError(
                    "Bidirectional S4D requires an even kernel channel count"
                )
            kernel = kernel.reshape(2, kernel.size(0) // 2, self.h, length)
            forward_kernel, reverse_kernel = kernel.unbind(0)
            kernel = F.pad(forward_kernel, (0, length)) + F.pad(
                reverse_kernel.flip(-1), (length, 0)
            )

        kernel_f = torch.fft.rfft(kernel, n=2 * length)
        input_f = torch.fft.rfft(u, n=2 * length)
        y = torch.einsum("bhl,chl->bchl", input_f, kernel_f)
        y = torch.fft.irfft(y, n=2 * length)[..., :length]
        y = y + torch.einsum("bhl,ch->bchl", u, self.D)
        y = y.flatten(1, 2)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


class SpikingSSMPathXBackbone(nn.Module):
    """Six-layer pre-norm residual backbone from SDN's ``SpikingSSM``."""

    def __init__(
        self,
        activation_factory: ActivationFactory,
        *,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 6,
        dropout: float = 0.0,
        prenorm: bool = True,
        norm: str = "batch",
        bidirectional: bool = True,
        layer_lr: float = 0.001,
        dt_min: float = 0.0001,
        dt_max: float = 0.1,
        kernel_factory: Optional[S4DKernelFactory] = None,
    ) -> None:
        super().__init__()
        norm = norm.lower()
        if norm not in {"batch", "layer"}:
            raise ValueError("norm must be 'batch' or 'layer'")

        self.prenorm = prenorm
        self.norm_kind = norm
        self.d_model = self.d_output = d_model
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                SpikingSSMS4DLayer(
                    d_model,
                    activation_factory,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=True,
                    bidirectional=bidirectional,
                    layer_lr=layer_lr,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    kernel_factory=kernel_factory,
                )
            )
            self.norms.append(
                nn.BatchNorm1d(d_model)
                if norm == "batch"
                else nn.LayerNorm(d_model)
            )
            self.dropouts.append(
                DropoutNd(dropout, tie=True, transposed=True)
                if dropout > 0.0
                else nn.Identity()
            )

    def _apply_norm(self, x: Tensor, norm: nn.Module) -> Tensor:
        if self.norm_kind == "batch":
            return norm(x)
        return norm(x.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, x: Tensor, **kwargs):
        """Map ``(B, L, H)`` to ``(B, L, H)`` with SDN residual blocks."""

        del kwargs
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected input shape (B, L, {self.d_model}), got {tuple(x.shape)}"
            )
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts
        ):
            residual = x
            z = self._apply_norm(x, norm) if self.prenorm else x
            z, _ = layer(z)
            z = dropout(z)
            x = residual + z
            if not self.prenorm:
                x = self._apply_norm(x, norm)
        return x.transpose(-1, -2), None


class SpikingSSMPathXClassifier(nn.Module):
    """Official-S4 PathX linear encoder/pool decoder around the SDN backbone."""

    def __init__(
        self,
        d_input: int,
        d_output: int,
        activation_factory: ActivationFactory,
        *,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 6,
        dropout: float = 0.0,
        prenorm: bool = True,
        norm: str = "batch",
        bidirectional: bool = True,
        layer_lr: float = 0.001,
        dt_min: float = 0.0001,
        dt_max: float = 0.1,
        sequence_length: int = 16384,
        kernel_factory: Optional[S4DKernelFactory] = None,
    ) -> None:
        super().__init__()
        self.config = SpikingSSMPathXConfig(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            norm=norm,
            bidirectional=bidirectional,
            layer_lr=layer_lr,
            dt_min=dt_min,
            dt_max=dt_max,
            sequence_length=sequence_length,
        )
        # The S4 training harness instantiates the backbone before its external
        # linear encoder and decoder; preserve that initialization order.
        self.backbone = SpikingSSMPathXBackbone(
            activation_factory,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            norm=norm,
            bidirectional=bidirectional,
            layer_lr=layer_lr,
            dt_min=dt_min,
            dt_max=dt_max,
            kernel_factory=kernel_factory,
        )
        self.encoder = nn.Linear(d_input, d_model)
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        if x.dim() != 3 or x.size(-1) != self.config.d_input:
            raise ValueError(
                "Expected continuous PathX input shape "
                f"(B, L, {self.config.d_input}), got {tuple(x.shape)}"
            )
        if lengths is not None and not torch.all(lengths == x.size(1)):
            raise ValueError("The SDN PathX recipe expects fixed-length sequences")

        x = self.encoder(x)
        x, _ = self.backbone(x)
        x = x.mean(dim=1)
        return self.decoder(x)


def build_spikingssm_pathx_dend_soma(
    *,
    d_input: int,
    d_output: int,
    vocab_size: Optional[int] = None,
    d_model: int = 256,
    d_state: int = 64,
    n_layers: int = 6,
    dropout: float = 0.0,
    sequence_length: int = 16384,
    dend_soma_num_branches: int = 8,
    dend_soma_compartments_per_branch: int = 4,
    dend_soma_branch_degree: int = 1,
    dend_soma_dend_backend: str = "fft",
    dend_soma_soma_type: str = "masked_sliding_psn",
    dend_soma_psn_order: Optional[int] = None,
    dend_soma_psn_backend: str = "fft",
    dend_soma_psn_exp_init: bool = True,
    dend_soma_psn_threshold_init: float = 0.0,
    dend_soma_ssf_thre: int = 4,
    dend_soma_activation_checkpoint: bool = True,
    kernel_factory: Optional[S4DKernelFactory] = None,
) -> SpikingSSMPathXClassifier:
    """Build the SDN PathX macro-model with this project's DEND+SOMA."""

    if vocab_size is not None:
        raise ValueError("PathX is a continuous-input task and cannot use an embedding")
    if dend_soma_psn_order is None:
        dend_soma_psn_order = sequence_length

    def activation_factory(width: int) -> nn.Module:
        return DendSomaS4Activation(
            width,
            transposed=True,
            num_branches=dend_soma_num_branches,
            compartments_per_branch=dend_soma_compartments_per_branch,
            branch_degree=dend_soma_branch_degree,
            dend_backend=dend_soma_dend_backend,
            soma_type=dend_soma_soma_type,
            psn_order=dend_soma_psn_order,
            psn_backend=dend_soma_psn_backend,
            psn_exp_init=dend_soma_psn_exp_init,
            psn_threshold_init=dend_soma_psn_threshold_init,
            ssf_thre=dend_soma_ssf_thre,
            activation_checkpoint=dend_soma_activation_checkpoint,
        )

    return SpikingSSMPathXClassifier(
        d_input=d_input,
        d_output=d_output,
        activation_factory=activation_factory,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        dropout=dropout,
        prenorm=True,
        norm="batch",
        bidirectional=True,
        layer_lr=0.001,
        dt_min=0.0001,
        dt_max=0.1,
        sequence_length=sequence_length,
        kernel_factory=kernel_factory,
    )


__all__ = [
    "SpikingSSMPathXBackbone",
    "SpikingSSMPathXClassifier",
    "SpikingSSMPathXConfig",
    "SpikingSSMS4DLayer",
    "build_spikingssm_pathx_dend_soma",
    "resolve_official_s4d_kernel",
]