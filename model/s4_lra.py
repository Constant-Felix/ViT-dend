"""Standard S4 backbone used as the LRA base model in MMDEND.

MMDEND reports that its LRA experiments follow the S4 architecture from
Gu et al. and replace the activation layer inside each of 6 S4 blocks with a
2-branch, 4-compartment MMDEND. This file keeps the original S4-style block by
default and can optionally replace the activation immediately after FFTConv
with the project's DEND+SOMA module. The output projection, GLU, residual, and
normalization remain unchanged from S4.

The public S4 repository exposes the paper-faithful optimized S4Block. If that
repository is on PYTHONPATH, use backend="official". The local fallback below
is a compact diagonal SSM layer with the same input/output interface. Select it
explicitly with backend="fallback" for shape tests and small experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Union

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


LayerLR = Union[float, Mapping[str, Optional[float]], None]
S4LayerFactory = Callable[..., nn.Module]


@dataclass
class S4LRAConfig:
    d_input: int
    d_output: int
    vocab_size: Optional[int] = None
    padding_idx: Optional[int] = None
    d_model: int = 256
    d_state: int = 64
    n_layers: int = 6
    dropout: float = 0.0
    tie_dropout: bool = True
    prenorm: bool = True
    norm: str = "batch"
    transposed: bool = False
    bidirectional: bool = True
    l_max: Optional[int] = None
    layer_lr: LayerLR = 0.001
    dt_min: float = 0.001
    dt_max: float = 0.1
    n_ssm: Optional[int] = None
    retrieval: bool = False
    use_dend_soma_activation: bool = False
    dend_soma_num_branches: int = 2
    dend_soma_compartments_per_branch: int = 4
    dend_soma_branch_degree: int = 2
    dend_soma_dend_backend: str = "fft"
    dend_soma_soma_type: str = "masked_sliding_psn"
    dend_soma_psn_order: Optional[int] = None
    dend_soma_psn_backend: str = "fft"
    dend_soma_psn_exp_init: bool = False
    dend_soma_psn_threshold_init: float = 0.0
    dend_soma_ssf_thre: int = 4
    dend_soma_activation_checkpoint: bool = True
    backend: str = "official"


def resolve_official_s4block() -> S4LayerFactory:
    """Return the optimized S4Block from HazyResearch/state-spaces."""

    try:
        from models.s4.s4 import S4Block
    except ImportError as exc:
        raise ImportError(
            "Could not import the official S4Block. Clone "
            "https://github.com/HazyResearch/state-spaces and add it to "
            "PYTHONPATH, or instantiate StandardS4ForLRA with backend='fallback'."
        ) from exc

    def factory(
        d_model: int,
        d_state: int,
        dropout: float,
        transposed: bool,
        *,
        tie_dropout: bool = False,
        bidirectional: bool = True,
        l_max: Optional[int] = None,
        layer_lr: LayerLR = 0.001,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        n_ssm: Optional[int] = None,
    ) -> nn.Module:
        return S4Block(
            d_model,
            l_max=l_max,
            d_state=d_state,
            dropout=dropout,
            tie_dropout=tie_dropout,
            transposed=transposed,
            lr=layer_lr,
            dt_min=dt_min,
            dt_max=dt_max,
            init="legs",
            bidirectional=bidirectional,
            n_ssm=n_ssm,
        )

    return factory


class DiagonalSSMLayer(nn.Module):
    """Small pure PyTorch SSM layer with the S4 block interface.

    This is not the optimized DPLR/Cauchy implementation from the S4 paper. It
    preserves the same tensor contract, ``(B, H, L) -> (B, H, L)`` when
    transposed=True, so the surrounding 6-block model can be built and tested
    without external dependencies.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed

        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_log = nn.Parameter(torch.full((d_model,), math.log(1e-3)))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
        )

    def _kernel(self, length: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        A = -torch.exp(self.A_log.float())
        dt = torch.exp(self.dt_log.float()).unsqueeze(-1)

        half_dt_A = 0.5 * dt * A
        A_bar = (1.0 + half_dt_A) / (1.0 - half_dt_A)
        B_bar = (dt * self.B.float()) / (1.0 - half_dt_A)

        powers = torch.arange(length, device=device, dtype=A_bar.dtype)
        powers = A_bar.unsqueeze(-1).pow(powers)
        kernel = (self.C.float() * B_bar).unsqueeze(-1) * powers
        return kernel.sum(dim=1).to(dtype=dtype)

    def forward(self, x: Tensor, state: Optional[Tensor] = None, **kwargs):
        if not self.transposed:
            x = x.transpose(-1, -2)

        length = x.size(-1)
        fft_length = 2 * length
        kernel = self._kernel(length, x.dtype, x.device)

        x_f = torch.fft.rfft(x, n=fft_length)
        k_f = torch.fft.rfft(kernel, n=fft_length)
        y = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=fft_length)[..., :length]
        y = y + x * self.D.view(1, -1, 1)

        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, state


def load_local_class(relative_path: str, module_name: str, class_name: str):
    """Load a class from a local file without importing the package __init__."""

    import importlib.util
    import sys
    from pathlib import Path

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        path = Path(__file__).resolve().parents[1] / relative_path
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {class_name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


class DropoutNd(nn.Module):
    """Dropout with an option to tie masks across the sequence dimension."""

    def __init__(self, p: float = 0.5, *, tie: bool = True, transposed: bool = False) -> None:
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if not self.tie:
            return nn.functional.dropout(x, p=self.p, training=True)

        if self.transposed:
            mask_shape = x.shape[:2] + (1,) * (x.dim() - 2)
        else:
            mask_shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
        mask = torch.rand(mask_shape, device=x.device) < 1.0 - self.p
        return x * mask.to(dtype=x.dtype) / (1.0 - self.p)


class DendSomaS4Activation(nn.Module):
    """Apply the channel-preserving dendrite and a selectable PSN soma."""

    def __init__(
        self,
        d_model: int,
        *,
        transposed: bool = False,
        num_branches: int = 2,
        compartments_per_branch: int = 4,
        branch_degree: int = 2,
        dend_backend: str = "fft",
        soma_type: str = "masked_sliding_psn",
        psn_order: int = 32,
        psn_backend: str = "fft",
        psn_exp_init: bool = False,
        psn_threshold_init: float = 0.0,
        ssf_thre: int = 4,
        activation_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        SparseChannelPreservingTrunkDistalDendCompartment = load_local_class(
            "module/dend_compartment.py",
            "_vit_dend_dend_compartment",
            "SparseChannelPreservingTrunkDistalDendCompartment",
        )

        soma_type = soma_type.lower()
        supported_soma_types = {"masked_sliding_psn", "psn_integer_ssf"}
        if soma_type not in supported_soma_types:
            choices = ", ".join(sorted(supported_soma_types))
            raise ValueError(f"soma_type must be one of: {choices}")
        if psn_order <= 0:
            raise ValueError("psn_order must be positive")
        if psn_backend not in {"gemm", "conv", "fft"}:
            raise ValueError("psn_backend must be 'gemm', 'conv', or 'fft'")
        if soma_type == "psn_integer_ssf" and ssf_thre <= 0:
            raise ValueError("ssf_thre must be positive")

        self.d_model = d_model
        self.transposed = transposed
        self.soma_type = soma_type
        self.activation_checkpoint = bool(activation_checkpoint)
        self.dend = SparseChannelPreservingTrunkDistalDendCompartment(
            channels=d_model,
            num_branches=num_branches,
            compartments_per_branch=compartments_per_branch,
            branch_degree=branch_degree,
            integration_backend=dend_backend,
            step_mode="m",
        )
        if self.soma_type == "masked_sliding_psn":
            MaskedSlidingPSN = load_local_class(
                "module/soma.py",
                "_vit_dend_soma",
                "MaskedSlidingPSN",
            )
            self.soma = MaskedSlidingPSN(
                order=psn_order,
                exp_init=psn_exp_init,
                backend=psn_backend,
            )
        else:
            PSNIntergerSoma_ssf = load_local_class(
                "module/soma.py",
                "_vit_dend_soma",
                "PSNIntergerSoma_ssf",
            )
            self.soma = PSNIntergerSoma_ssf(
                psn_order=psn_order,
                psn_exp_init=psn_exp_init,
                psn_backend=psn_backend,
                psn_threshold_init=psn_threshold_init,
                thre=ssf_thre,
                step_mode="m",
            )

    def _reset_state(self) -> None:
        if hasattr(self.dend, "reset"):
            self.dend.reset()
        if hasattr(self.soma, "reset"):
            self.soma.reset()

    def _forward_time_first(self, x_seq: Tensor) -> Tensor:
        # Reset inside the checkpointed region so backward recomputation starts
        # from the same zero-state trajectory as the original forward pass.
        self._reset_state()
        return self.soma(self.dend(x_seq))

    def _apply_time_first(self, x_seq: Tensor) -> Tensor:
        if (
            self.activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                self._forward_time_first,
                x_seq,
                use_reentrant=False,
            )
        return self._forward_time_first(x_seq)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D S4 activation input, got {tuple(x.shape)}")

        if self.transposed:
            if x.size(1) != self.d_model:
                raise ValueError(
                    f"Expected transposed input shape (B, {self.d_model}, L), "
                    f"got {tuple(x.shape)}"
                )
            x_seq = x.permute(2, 0, 1).contiguous()
            y_seq = self._apply_time_first(x_seq)
            return y_seq.permute(1, 2, 0).contiguous()

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected input shape (B, L, {self.d_model}), got {tuple(x.shape)}"
            )
        x_seq = x.transpose(0, 1).contiguous()
        y_seq = self._apply_time_first(x_seq)
        return y_seq.transpose(0, 1).contiguous()


def replace_s4_activation(
    s4_module: nn.Module,
    activation: nn.Module,
) -> None:
    """Replace only the activation immediately after the S4/FFTConv layer."""

    if hasattr(s4_module, "layer") and hasattr(s4_module.layer, "activation"):
        s4_module.layer.activation = activation
    elif hasattr(s4_module, "activation"):
        s4_module.activation = activation
    else:
        raise AttributeError(
            "Could not find the main S4 activation. Expected either "
            "`s4.layer.activation` or `s4.activation`."
        )


class StandardS4Block(nn.Module):
    """Residual S4 block used by the LRA classifier.

    The default input and output shape are both ``(B, L, H)``; set
    ``transposed=True`` for ``(B, H, L)``. For MMDEND-style changes, the
    activation to replace lives inside ``self.s4``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        tie_dropout: bool = False,
        prenorm: bool = True,
        norm: str = "batch",
        transposed: bool = False,
        bidirectional: bool = True,
        l_max: Optional[int] = None,
        layer_lr: LayerLR = 0.001,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        n_ssm: Optional[int] = None,
        use_dend_soma_activation: bool = False,
        dend_soma_num_branches: int = 2,
        dend_soma_compartments_per_branch: int = 4,
        dend_soma_branch_degree: int = 2,
        dend_soma_dend_backend: str = "fft",
        dend_soma_soma_type: str = "masked_sliding_psn",
        dend_soma_psn_order: Optional[int] = None,
        dend_soma_psn_backend: str = "fft",
        dend_soma_psn_exp_init: bool = False,
        dend_soma_psn_threshold_init: float = 0.0,
        dend_soma_ssf_thre: int = 4,
        dend_soma_activation_checkpoint: bool = True,
        s4_layer_factory: Optional[S4LayerFactory] = None,
    ) -> None:
        super().__init__()
        self.prenorm = prenorm
        self.norm_kind = norm.lower()
        self.transposed = transposed

        if use_dend_soma_activation and dend_soma_psn_order is None:
            if l_max is None:
                raise ValueError(
                    "dend_soma_psn_order must be set when the sequence length "
                    "l_max is unknown"
                )
            dend_soma_psn_order = l_max

        if s4_layer_factory is None:
            s4_layer_factory = DiagonalSSMLayer
        self.s4 = s4_layer_factory(
            d_model,
            d_state,
            dropout,
            transposed,
            tie_dropout=tie_dropout,
            bidirectional=bidirectional,
            l_max=l_max,
            layer_lr=layer_lr,
            dt_min=dt_min,
            dt_max=dt_max,
            n_ssm=n_ssm,
        )
        if use_dend_soma_activation:
            activation_transposed = isinstance(self.s4, DiagonalSSMLayer)
            activation_kwargs = {
                "transposed": activation_transposed,
                "num_branches": dend_soma_num_branches,
                "compartments_per_branch": dend_soma_compartments_per_branch,
                "branch_degree": dend_soma_branch_degree,
                "dend_backend": dend_soma_dend_backend,
                "soma_type": dend_soma_soma_type,
                "psn_order": dend_soma_psn_order,
                "psn_backend": dend_soma_psn_backend,
                "psn_exp_init": dend_soma_psn_exp_init,
                "psn_threshold_init": dend_soma_psn_threshold_init,
                "ssf_thre": dend_soma_ssf_thre,
                "activation_checkpoint": dend_soma_activation_checkpoint,
            }
            replace_s4_activation(
                self.s4,
                activation=DendSomaS4Activation(
                    d_model,
                    **activation_kwargs,
                ),
            )

        self.dropout = (
            DropoutNd(dropout, tie=True, transposed=transposed)
            if tie_dropout and dropout > 0.0
            else nn.Dropout(dropout)
        )

        if self.norm_kind == "batch":
            self.norm = nn.BatchNorm1d(d_model)
        elif self.norm_kind == "layer":
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    def _apply_norm(self, x: Tensor) -> Tensor:
        if self.norm_kind == "batch":
            if self.transposed:
                return self.norm(x)
            return self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        if self.transposed:
            return self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        return self.norm(x)

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        residual = x
        z = self._apply_norm(x) if self.prenorm else x

        z, _ = self.s4(z, lengths=lengths)
        z = self.dropout(z)
        x = residual + z

        if not self.prenorm:
            x = self._apply_norm(x)
        return x


class RetrievalHead(nn.Module):
    """Official S4/LRA NLI-style decoder for paired AAN documents."""

    def __init__(self, d_input: int, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4 * d_input, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.size(0) % 2 != 0:
            raise ValueError(
                "AAN retrieval expects the official S4 collate order with a 2B batch"
            )
        first, second = x.reshape(2, x.size(0) // 2, x.size(-1)).unbind(0)
        features = torch.cat(
            [first, second, first - second, first * second], dim=-1
        )
        return self.classifier(features)


class StandardS4ForLRA(nn.Module):
    """6-block S4 sequence classifier for LRA-style tasks.

    Continuous inputs use shape ``(B, L, d_input)``. Token inputs set
    ``vocab_size`` and use shape ``(B, L)``.
    Sequence length ``L`` is pooled by mean pooling after the final S4 block.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        vocab_size: Optional[int] = None,
        padding_idx: Optional[int] = None,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 6,
        dropout: float = 0.0,
        tie_dropout: bool = True,
        prenorm: bool = True,
        norm: str = "batch",
        transposed: bool = False,
        bidirectional: bool = True,
        l_max: Optional[int] = None,
        layer_lr: LayerLR = 0.001,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        n_ssm: Optional[int] = None,
        retrieval: bool = False,
        use_dend_soma_activation: bool = False,
        dend_soma_num_branches: int = 2,
        dend_soma_compartments_per_branch: int = 4,
        dend_soma_branch_degree: int = 2,
        dend_soma_dend_backend: str = "fft",
        dend_soma_soma_type: str = "masked_sliding_psn",
        dend_soma_psn_order: Optional[int] = None,
        dend_soma_psn_backend: str = "fft",
        dend_soma_psn_exp_init: bool = False,
        dend_soma_psn_threshold_init: float = 0.0,
        dend_soma_ssf_thre: int = 4,
        dend_soma_activation_checkpoint: bool = True,
        backend: str = "official",
    ) -> None:
        super().__init__()
        if use_dend_soma_activation and dend_soma_psn_order is None:
            if l_max is None:
                raise ValueError(
                    "dend_soma_psn_order must be set when the sequence length "
                    "l_max is unknown"
                )
            dend_soma_psn_order = l_max
        self.vocab_size = vocab_size
        self.transposed = transposed
        self.prenorm = prenorm
        self.norm_kind = norm.lower()
        self.config = S4LRAConfig(
            d_input=d_input,
            d_output=d_output,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            tie_dropout=tie_dropout,
            prenorm=prenorm,
            norm=norm,
            transposed=transposed,
            bidirectional=bidirectional,
            l_max=l_max,
            layer_lr=layer_lr,
            dt_min=dt_min,
            dt_max=dt_max,
            n_ssm=n_ssm,
            retrieval=retrieval,
            use_dend_soma_activation=use_dend_soma_activation,
            dend_soma_num_branches=dend_soma_num_branches,
            dend_soma_compartments_per_branch=dend_soma_compartments_per_branch,
            dend_soma_branch_degree=dend_soma_branch_degree,
            dend_soma_dend_backend=dend_soma_dend_backend,
            dend_soma_soma_type=dend_soma_soma_type,
            dend_soma_psn_order=dend_soma_psn_order,
            dend_soma_psn_backend=dend_soma_psn_backend,
            dend_soma_psn_exp_init=dend_soma_psn_exp_init,
            dend_soma_psn_threshold_init=dend_soma_psn_threshold_init,
            dend_soma_ssf_thre=dend_soma_ssf_thre,
            dend_soma_activation_checkpoint=dend_soma_activation_checkpoint,
            backend=backend,
        )

        if vocab_size is None:
            self.encoder = nn.Linear(d_input, d_model)
        else:
            self.encoder = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        s4_layer_factory = self._select_s4_factory(backend)
        self.blocks = nn.ModuleList(
            [
                StandardS4Block(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    tie_dropout=tie_dropout,
                    prenorm=prenorm,
                    norm=norm,
                    transposed=transposed,
                    bidirectional=bidirectional,
                    l_max=l_max,
                    layer_lr=layer_lr,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    n_ssm=n_ssm,
                    use_dend_soma_activation=use_dend_soma_activation,
                    dend_soma_num_branches=dend_soma_num_branches,
                    dend_soma_compartments_per_branch=dend_soma_compartments_per_branch,
                    dend_soma_branch_degree=dend_soma_branch_degree,
                    dend_soma_dend_backend=dend_soma_dend_backend,
                    dend_soma_soma_type=dend_soma_soma_type,
                    dend_soma_psn_order=dend_soma_psn_order,
                    dend_soma_psn_backend=dend_soma_psn_backend,
                    dend_soma_psn_exp_init=dend_soma_psn_exp_init,
                    dend_soma_psn_threshold_init=dend_soma_psn_threshold_init,
                    dend_soma_ssf_thre=dend_soma_ssf_thre,
                    dend_soma_activation_checkpoint=dend_soma_activation_checkpoint,
                    s4_layer_factory=s4_layer_factory,
                )
                for _ in range(n_layers)
            ]
        )
        if self.norm_kind == "batch":
            self.final_norm = nn.BatchNorm1d(d_model) if prenorm else nn.Identity()
        elif self.norm_kind == "layer":
            self.final_norm = nn.LayerNorm(d_model) if prenorm else nn.Identity()
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        self.decoder = (
            RetrievalHead(d_model, d_model, d_output)
            if retrieval
            else nn.Linear(d_model, d_output)
        )

    @staticmethod
    def _select_s4_factory(backend: str) -> S4LayerFactory:
        backend = backend.lower()
        if backend == "official":
            return resolve_official_s4block()
        if backend == "fallback":
            return DiagonalSSMLayer
        if backend == "auto":
            try:
                return resolve_official_s4block()
            except ImportError:
                return DiagonalSSMLayer
        raise ValueError("backend must be one of: 'auto', 'official', 'fallback'")

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        if self.vocab_size is None:
            if x.dim() != 3:
                raise ValueError(f"Expected input shape (B, L, d_input), got {tuple(x.shape)}")
            x = self.encoder(x)
        else:
            if x.dim() != 2:
                raise ValueError(f"Expected token input shape (B, L), got {tuple(x.shape)}")
            x = self.encoder(x.long())

        if self.transposed:
            x = x.transpose(-1, -2)

        for block in self.blocks:
            x = block(x, lengths=lengths)

        if isinstance(self.final_norm, nn.BatchNorm1d):
            if self.transposed:
                x = self.final_norm(x)
            else:
                x = self.final_norm(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            if self.transposed:
                x = self.final_norm(x.transpose(-1, -2)).transpose(-1, -2)
            else:
                x = self.final_norm(x)

        if self.transposed:
            x = x.transpose(-1, -2)
        if lengths is None:
            x = x.mean(dim=1)
        else:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = (x * mask.unsqueeze(-1)).sum(dim=1)
            x = x / lengths.clamp_min(1).unsqueeze(-1)

        return self.decoder(x)


LRA_S4_PRESETS: Dict[str, Dict[str, object]] = {
    "listops": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 4,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": False,
        "l_max": 2048,
        "layer_lr": {"dt": None, "A": 0.001, "B": 0.001},
        "n_ssm": 1,
    },
    "text": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 4,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": True,
        "l_max": 4096,
        "layer_lr": {"dt": None, "A": 0.001, "B": 0.001},
        "n_ssm": 1,
    },
    "imdb": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 4,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": True,
        "l_max": 4096,
        "layer_lr": {"dt": None, "A": 0.001, "B": 0.001},
        "n_ssm": 1,
    },
    "aan": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 4,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": True,
        "l_max": 4000,
        "layer_lr": {"dt": None, "A": 0.001, "B": 0.001},
        "n_ssm": 1,
        "retrieval": True,
    },
    "retrieval": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 4,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": True,
        "l_max": 4000,
        "layer_lr": {"dt": None, "A": 0.001, "B": 0.001},
        "n_ssm": 1,
        "retrieval": True,
    },
    "image": {
        "n_layers": 6,
        "d_model": 512,
        "d_state": 64,
        "dropout": 0.1,
        "tie_dropout": True,
        "norm": "batch",
        "prenorm": False,
        "l_max": 1024,
        "n_ssm": 1,
    },
    "cifar": {
        "n_layers": 6,
        "d_model": 512,
        "d_state": 64,
        "dropout": 0.1,
        "tie_dropout": True,
        "norm": "batch",
        "prenorm": False,
        "l_max": 1024,
        "n_ssm": 1,
    },
    "pathfinder": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 64,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": True,
        "l_max": 1024,
        "n_ssm": None,
    },
    "pathx": {
        "n_layers": 6,
        "d_model": 256,
        "d_state": 64,
        "dropout": 0.0,
        "norm": "batch",
        "prenorm": True,
        "l_max": 16384,
        "dt_min": 0.0001,
        "n_ssm": None,
    },
}


def build_standard_s4_lra(
    task: str,
    d_input: int,
    d_output: int,
    vocab_size: Optional[int] = None,
    backend: str = "official",
    **overrides,
) -> StandardS4ForLRA:
    """Build the S4 backbone with MMDEND/S4-repo LRA task defaults."""

    config = dict(LRA_S4_PRESETS.get(task.lower(), {}))
    config.update(overrides)
    #config.setdefault("n_layers", 6)
    if config.get("n_ssm") == "d_model":
        config["n_ssm"] = config["d_model"]
    return StandardS4ForLRA(
        d_input=d_input,
        d_output=d_output,
        vocab_size=vocab_size,
        backend=backend,
        **config,
    )


def build_dend_soma_s4_lra(
    task: str,
    d_input: int,
    d_output: int,
    vocab_size: Optional[int] = None,
    backend: str = "official",
    **overrides,
) -> StandardS4ForLRA:
    """Build LRA S4 with DEND+SOMA after FFTConv in each S4 block."""

    overrides.setdefault("use_dend_soma_activation", True)
    return build_standard_s4_lra(
        task=task,
        d_input=d_input,
        d_output=d_output,
        vocab_size=vocab_size,
        backend=backend,
        **overrides,
    )

