"""The voltage dynamics of dendritic compartments.

This package contains a series of classes depicting different types of dendritic
compartments so that we can compute the dendritic voltage dynamics step by step,
given the input to all the compartments. When computing the dendritic voltage 
dynamics for a single time step, the compartments are treated independently. 
The relationship (wiring) among a set of compartments is not considered here.
"""

import abc
from typing import Callable
import torch.nn.functional as F
import torch
import torch.nn as nn
from spikingjelly.activation_based import base


class BaseDendCompartment(base.MemoryModule, abc.ABC):
    """Base class for all dendritic compartments.

    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
        store_v_seq (bool): whether to store the compartmental potential at 
            every time step when using multi-step mode. If True, there is 
            another attribute called v_seq.
    """

    def __init__(
        self, v_init: float = 0., 
        step_mode: str = "s", store_v_seq: bool = False,
    ):
        """The constructor of BaseDendCompartment.

        Args:
            v_init (float, optional): initial voltage (at time step 0). 
                Defaults to 0..
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
            store_v_seq (bool, optional): whether to store the compartmental 
                potential at every time step when using multi-step mode. 
                Defaults to False.
        """
        super().__init__()
        self.register_memory("v", v_init)
        self.step_mode = step_mode
        self.store_v_seq = store_v_seq

    @property
    def store_v_seq(self) -> bool:
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, val: bool):
        self._store_v_seq = val
        if val and (not hasattr(self, "v_seq")):
            self.register_memory("v_seq", None)

    def v_float2tensor(self, x: torch.Tensor):
        """If self.v is a float, turn it into a tensor with x's shape.
        """
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

    @abc.abstractmethod
    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(y_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        elif self.step_mode == "m":
            return self.multi_step_forward(x)
        else:
            raise ValueError(
                f"BaseDendCompartment.step_mode shoud be 'm' or 's', "
                f"but get {self.step_mode} instead."
            )


class PassiveDendCompartment(BaseDendCompartment):
    """
    Passive dendritic compartment with learnable SCALAR tau.
    
    Fixed:
    1. Causal Matrix Construction (Lower Triangular).
    2. Supports input shape (T, B, C_sub, H, W, N).
    3. Tau is shared across all dimensions.
    """

    def __init__(
        self,
        num_branches = 2,
        c_sub=1,
        tau = 2.0, #
        soma_dim = 3,
        decay_input: bool = True,  ###
        gating = True,
        bn = True,
        #bn_alter = False,
        res = False,
        v_rest: float = 0.0,
        #v_init: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter = False,
        use_astro: bool = True, 
        #use_oli: bool = True
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        self.soma_dim = soma_dim
        self.gating = gating
        self.bn = bn
        #self.bn_alter = bn_alter
        self.skip_weight = nn.Parameter(torch.tensor(0.01))
        self.res = res
        self.no_filter = no_filter
        self.use_astro = use_astro
        #self.use_oli = use_oli
        self.c_sub = c_sub
        self.num_branches = num_branches
        self.astro_bias = nn.Parameter(torch.full((1, 1, 1, 1, num_branches), 0.0))  ##

        tau_data = torch.full((num_branches,), float(tau))
        
        self.tau_branches = nn.Parameter(torch.tensor(tau_data, dtype=torch.float32))    
        self.decay_input = decay_input
        self.v_rest = v_rest

        if self.use_astro:
                # [星形胶质细胞] 慢速积分状态
                # 钙离子衰减系数 (0.9 ~ 0.99)
            self.astro_decay = nn.Parameter(torch.tensor(0.9))
                # 钙离子对门控的影响力 (Gain)
            self.astro_gain = nn.Parameter(torch.tensor(1.0))
                # 内部状态
            self.ca_state = 0.0 

        if self.gating:
            if self.bn:
                if self.soma_dim == 3:
                # BN + Gating 模式 (静态图/DVS通用)
                    self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                    self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches))
                else:
                    self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, num_branches))
                    self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, num_branches))

                self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) if self.soma_dim == 3 else nn.BatchNorm1d(self.c_sub, affine=True, momentum=0.1)  
                nn.init.constant_(self.branch_bn.weight, 1.0) ## 1.5
                nn.init.constant_(self.branch_bn.bias, 0.0) ## 0.5
            else:
                self.gate_scale = nn.Parameter(torch.tensor(1.0))
                self.gate_beta2 = nn.Parameter(torch.tensor(0.0))
        
    def _build_tau_terms(self, T: int, device, dtype):
        """
        现在 tau_branches 是 (N,) 的张量。
        为了能使用矩阵乘法，我们要对 N 个分支分别构建积分矩阵。
        """
        # 限制最小 tau，防止衰减为 0
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-5) # 形状: (N,)
        a = 1.0 - 1.0 / tau # 形状: (N,)

        t = torch.arange(T, device=device, dtype=dtype)
        i = t[:, None]
        j = t[None, :]
        diff = i - j
        mask = diff >= 0
        
        # 构建 tau_matrix，形状将会是 (N, T, T)
        # 这意味着 N 个分支各有自己的一套下三角积分矩阵
        tau_matrix = torch.zeros((len(tau), T, T), device=device, dtype=dtype)
        
        for n in range(len(tau)):
            tau_matrix[n] = tau_matrix[n].masked_scatter(mask, (a[n] ** diff[mask]).to(dtype))

        # 系数 v_init 和 v_rest 也会变成 (N, T)
        tau_vec_init = a.unsqueeze(1) ** (t + 1.0) # (N, T)
        tau_vec_rest = 1.0 - tau_vec_init # (N, T)

        return tau_matrix, tau_vec_init, tau_vec_rest, tau
    
    
    def _compute_astro_modulation(self, y_seq):
        """
        计算星形胶质细胞的钙信号，独立调节每个分支 (N) 的强度。
        """
        if not self.use_astro:
            return 1.0 # 无调节
            
        T = y_seq.shape[0]
        B = y_seq.shape[1]
        N = y_seq.shape[-1]  # [关键修改 1]: 动态获取分支数 N
        device = y_seq.device
        
        # 1. 能量输入 (取绝对值)
        # [关键修改 2]: 去掉在维度 N 上的均值操作
        if self.soma_dim == 3:
            # y_seq: (T, B, C, H, W, N) -> 对 C(2), H(3), W(4) 求均值，保留 N(5)
            energy_seq = y_seq.abs().mean(dim=(2, 3, 4)) # 结果 shape: (T, B, N)
        else:
            # y_seq: (T, B, C, H, N) -> 对 C(2), H(3) 求均值，保留 N(4)
            energy_seq = y_seq.abs().mean(dim=(2, 3))    # 结果 shape: (T, B, N)
        
        # 2. 慢速积分 (Leaky Integration)
        ca_seq = []
        
        # [关键修改 3]: 初始化时给每个样本的每个分支分配独立的钙状态
        curr_ca = torch.zeros((B, N), device=device)

        decay = torch.clamp(self.astro_decay, 0.0, 1.0)
        
        for t in range(T):
            curr_ca = decay * curr_ca + (1 - decay) * torch.tanh(energy_seq[t])
            ca_seq.append(curr_ca)
            
        # 更新状态
        self.ca_state = curr_ca.detach()
        
        # 3. 生成调节系数 (Modulation Factor)
        # ca_signal 的 shape 是 (T, B, N)
        ca_signal = torch.stack(ca_seq, dim=0)
        
        # [关键修改 4]: 巧妙重塑形状，让 Pytorch 的广播机制发挥作用
        if self.soma_dim == 3:
            # 变回 (T, B, 1, 1, 1, N) 从而能与 (T, B, C, H, W, N) 完美相乘
            ca_signal = ca_signal.view(T, B, 1, 1, 1, N)
        else:
            # 变回 (T, B, 1, 1, N) 从而能与 (T, B, C, H, N) 完美相乘
            ca_signal = ca_signal.view(T, B, 1, 1, N)    
        
        #ca_baseline = ca_signal.mean(dim=(0, 1), keepdim=True) 
        #modulation = 1.0 + 0.5 * torch.tanh(self.astro_gain * (ca_signal - ca_baseline))
        modulation = 1.0 + 0.5 * torch.tanh(self.astro_gain * ca_signal + self.astro_bias)

        return modulation

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-3)

        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        if self.decay_input:
            v = self.single_step_decay_input(self.v, x, self.v_rest, tau)
        else:
            v = self.single_step_not_decay_input(self.v, x, self.v_rest, tau)

        # IMPORTANT:
        # detach state to avoid cross-iteration graph leakage
        self.v = v.detach()
        return v

    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: Shape (T, B, C_sub, H, W, N)
        Returns:
            y: Shape (T, B, C_sub, H, W, N)
        """
        if self.soma_dim==3:
            T, B, C_sub, H, W, N = x_seq.shape # T,B,C/N,H,W,N
        else:
            T, B, C_sub, H, N = x_seq.shape  # W,B,C/N,H,N
        device = x_seq.device
        dtype = x_seq.dtype
        
        if self.no_filter == False:

        
            (tau_matrix, tau_vec_init, tau_vec_rest, tau) = \
            self._build_tau_terms(T, device, dtype)

            if self.decay_input:
                
                if self.soma_dim == 3:
                    div_tau = tau.view(1, 1, 1, 1, 1, N)
                else:
                    div_tau = tau.view(1, 1, 1, 1, N)    
                x_seq = x_seq / div_tau

            y_branches = []
            for n in range(self.num_branches):
                x_n = x_seq[..., n].reshape(T, -1)
                y_n_flat = torch.matmul(tau_matrix[n], x_n)
                if self.soma_dim == 3:
                    y_n = y_n_flat.view(T, B, C_sub, H, W)
                else:
                    y_n = y_n_flat.view(T, B, C_sub, H)    
                
                # Init & Rest
                t_init = tau_vec_init[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_init[n].view(T, 1, 1, 1)
                t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_rest[n].view(T, 1, 1, 1)
                
                v_init = self.v
                if isinstance(v_init, torch.Tensor):
                    v_in = v_init[..., n].detach().unsqueeze(0)
                else:
                    v_in = v_init
                    
                y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
                y_branches.append(y_n)

            y = torch.stack(y_branches, dim=-1) # (T, B, ..., N)
            
        
        else:
            y = x_seq
              
        if self.store_v_seq:
            self.v_seq = y

        # Detach and store last state
        # State shape: (B, C_sub, H, W, N)
        self.v = y[-1].detach()
        
        if self.gating:

            astro_mod = self._compute_astro_modulation(y)
            y = y * astro_mod
            if self.bn:
                # [模式 A: BN + Sigmoid Gate]
                # 这是最强大的组合：BN 提供标准化，Gating 提供非线性，Astro 提供上下文
                
                if self.soma_dim == 3:
                        y_permuted = y.permute(0, 1, 5, 2, 3, 4).contiguous()
                    
                        # 2. Reshape into standard 2D Convolution format for BN
                        # batch_size = T*B, channels = N*C_sub
                        y_reshaped = y_permuted.view(T * B, N * C_sub, H, W)
                        
                        # 3. Apply BatchNorm2d (各分支、各通道拥有独立的 \gamma 和 \beta)
                        y_normed_reshaped = self.branch_bn(y_reshaped)
                        
                        # 4. Restore shape
                        y_normed = y_normed_reshaped.view(T, B, N, C_sub, H, W)
                        # (T, B, N, C_sub, H, W) -> permute back -> (T, B, C_sub, H, W, N)
                        y_normed = y_normed.permute(0, 1, 3, 4, 5, 2).contiguous()

                else:
                        y_permuted = y.permute(0, 1, 4, 2, 3).contiguous()
                        
                        # 2. 融合维度：(Batch, Channels, Length)
                        # 目标形状: (T * B, C * N, H)
                        y_reshaped = y_permuted.view(T * B, N * C_sub, H)
                        
                        # 3. 过 BatchNorm1d
                        y_normed_reshaped = self.branch_bn(y_reshaped)
                        
                        # 4. 还原形状：先解开 view -> (T, B, C, N, H)
                        y_normed = y_normed_reshaped.view(T, B, N, C_sub, H)
                        
                        # 5. 再把 N 移回最后 -> (T, B, C, H, N)
                        y_normed = y_normed.permute(0, 1, 3, 4, 2).contiguous()        
            
                
                # 最后的非线性门控
                gate = torch.sigmoid(self.gate_alpha * y_normed + self.gate_beta)
                y_out = y_normed * gate
                
            else:
                gate_factor = torch.sigmoid(self.gate_scale * y + self.gate_beta2)
                y_out = y * gate_factor    
        else:
            y_out = y


        if self.res == True:
            y_out = y_out + self.skip_weight * x_seq
          
               
            
        return y_out
        
class MultiScaleDendCompartment(BaseDendCompartment):
    """
    Passive dendritic compartment with learnable SCALAR tau.
    
    Fixed:
    1. Causal Matrix Construction (Lower Triangular).
    2. Supports input shape (T, B, C_sub, H, W, N).
    3. Tau is shared across all dimensions.
    """

    def __init__(
        self,
        num_branches,
        c_sub=1,
        init_tau = [1.25,10.0], 
        soma_dim = 3,
        decay_input: bool = True,  ###
        gating = True,
        bn = True,  ##
        res = False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter = False,
        use_astro: bool = True, 
        last_sigmoid = False,
        last_tanh = True,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        self.soma_dim = soma_dim
        self.gating = gating
        self.bn = bn
        #self.bn_alter = bn_alter
        self.skip_weight = nn.Parameter(torch.tensor(0.01))
        self.res = res
        self.no_filter = no_filter
        self.use_astro = use_astro
        self.last_sigmoid = last_sigmoid
        self.last_tanh = last_tanh
        #self.use_oli = use_oli
        self.c_sub = c_sub
        self.num_branches = num_branches
        self.astro_bias = nn.Parameter(torch.full((1, 1, 1, 1, num_branches), -1.0))
        #self.alpha = nn.Parameter(torch.tensor(0.7))
        #self.beta = nn.Parameter(torch.tensor(1.0))
        #self.w_b = nn.Parameter(torch.tensor(0.1))
        if init_tau is None:
            tau_data = torch.linspace(2.0, 10.0, steps=num_branches)
        elif isinstance(init_tau, (float, int)):
            tau_data = torch.full((num_branches,), float(init_tau))
        else:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)

        self.tau_branches = nn.Parameter(torch.tensor(tau_data, dtype=torch.float32))    
        self.decay_input = decay_input
        self.v_rest = v_rest

        #if self.use_oli:
                # [少突胶质细胞] 动态 Tau 调节系数
                # 髓鞘化灵敏度，初始化为 0.5
            #self.oligo_alpha = nn.Parameter(torch.tensor(0.5))

        if self.use_astro:
                # [星形胶质细胞] 慢速积分状态
                # 钙离子衰减系数 (0.9 ~ 0.99)
            self.astro_decay = nn.Parameter(torch.tensor(0.9))
                # 钙离子对门控的影响力 (Gain)
            self.astro_gain = nn.Parameter(torch.tensor(1.0))
                # 内部状态
            self.ca_state = 0.0 

        if self.gating:
            if self.bn:
                if last_sigmoid == True or last_tanh == True:
                    if self.soma_dim == 3:
                    # BN + Gating 模式 (静态图/DVS通用)
                        self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                        self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches)) #if last_sigmoid == True else nn.Parameter(torch.full((1, 1, 1, 1, 1, num_branches), -1.0))
                    else:
                        self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, num_branches))
                        self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, num_branches)) #if last_sigmoid == True else nn.Parameter(torch.full((1, 1, 1, 1, num_branches), -1.0))

                self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) if self.soma_dim == 3 else nn.BatchNorm1d(self.c_sub, affine=True, momentum=0.1)  
                nn.init.constant_(self.branch_bn.weight, 1.0) ## 1.5
                nn.init.constant_(self.branch_bn.bias, 0.0) ## 0.5
            else:
                if last_sigmoid == True or last_tanh == True:
                    self.gate_scale = nn.Parameter(torch.tensor(1.0))
                    self.gate_beta2 = nn.Parameter(torch.tensor(0.0))
        
    def _build_tau_terms(self, T: int, device, dtype):
        """
        现在 tau_branches 是 (N,) 的张量。
        为了能使用矩阵乘法，我们要对 N 个分支分别构建积分矩阵。
        """
        # 限制最小 tau，防止衰减为 0
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-5) # 形状: (N,)
        a = 1.0 - 1.0 / tau # 形状: (N,)

        t = torch.arange(T, device=device, dtype=dtype)
        i = t[:, None]
        j = t[None, :]
        diff = i - j
        mask = diff >= 0
        
        # 构建 tau_matrix，形状将会是 (N, T, T)
        # 这意味着 N 个分支各有自己的一套下三角积分矩阵
        tau_matrix = torch.zeros((len(tau), T, T), device=device, dtype=dtype)
        
        for n in range(len(tau)):
            tau_matrix[n] = tau_matrix[n].masked_scatter(mask, (a[n] ** diff[mask]).to(dtype))

        # 系数 v_init 和 v_rest 也会变成 (N, T)
        tau_vec_init = a.unsqueeze(1) ** (t + 1.0) # (N, T)
        tau_vec_rest = 1.0 - tau_vec_init # (N, T)

        return tau_matrix, tau_vec_init, tau_vec_rest, tau
    
    
    def _compute_astro_modulation(self, y_seq):
        """
        计算星形胶质细胞的钙信号，独立调节每个分支 (N) 的强度。
        """
        if not self.use_astro:
            return 1.0 # 无调节
            
        T = y_seq.shape[0]
        B = y_seq.shape[1]
        N = y_seq.shape[-1]  # [关键修改 1]: 动态获取分支数 N
        device = y_seq.device
        
        # 1. 能量输入 (取绝对值)
        # [关键修改 2]: 去掉在维度 N 上的均值操作

        if self.soma_dim == 3:
            # y_seq: (T, B, C, H, W, N) -> 对 C(2), H(3), W(4) 求均值，保留 N(5)
            energy_seq = y_seq.abs().mean(dim=(2, 3, 4)) # 结果 shape: (T, B, N)
        else:
            # y_seq: (T, B, C, H, N) -> 对 C(2), H(3) 求均值，保留 N(4)
            energy_seq = y_seq.abs().mean(dim=(2, 3))    # 结果 shape: (T, B, N)
        
        # 2. 慢速积分 (Leaky Integration)
        ca_seq = []
        
        # [关键修改 3]: 初始化时给每个样本的每个分支分配独立的钙状态
        curr_ca = torch.zeros((B, N), device=device)

        decay = torch.clamp(self.astro_decay, 0.0, 1.0)
        
        for t in range(T):
            curr_ca = decay * curr_ca + (1 - decay) * energy_seq[t]  ## 加了tanh
            ca_seq.append(curr_ca)
            
        # 更新状态
        self.ca_state = curr_ca.detach()
         
        # 3. 生成调节系数 (Modulation Factor)
        # ca_signal 的 shape 是 (T, B, N)
        ca_signal = torch.stack(ca_seq, dim=0)
        
        # [关键修改 4]: 巧妙重塑形状，让 Pytorch 的广播机制发挥作用
        if self.soma_dim == 3:
            # 变回 (T, B, 1, 1, 1, N) 从而能与 (T, B, C, H, W, N) 完美相乘
            ca_signal = ca_signal.view(T, B, 1, 1, 1, N)
        else:
            # 变回 (T, B, 1, 1, N) 从而能与 (T, B, C, H, N) 完美相乘
            ca_signal = ca_signal.view(T, B, 1, 1, N)    
        
        #ca_baseline = ca_signal.mean(dim=(0, 1), keepdim=True) 
        #modulation = 1.0 + torch.tanh(self.astro_gain * (ca_signal - ca_baseline))
        #add = torch.tanh(self.w_b * ca_signal) # (B, N)
        modulation = 1.0 + 0.5 * torch.tanh(self.astro_gain * ca_signal + self.astro_bias)

        return modulation

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-3)

        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        if self.decay_input:
            v = self.single_step_decay_input(self.v, x, self.v_rest, tau)
        else:
            v = self.single_step_not_decay_input(self.v, x, self.v_rest, tau)

        # IMPORTANT:
        # detach state to avoid cross-iteration graph leakage
        self.v = v.detach()
        return v

    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: Shape (T, B, C_sub, H, W, N)
        Returns:
            y: Shape (T, B, C_sub, H, W, N)
        """
        if self.soma_dim==3:
            T, B, C_sub, H, W, N = x_seq.shape # T,B,C/N,H,W,N
        else:
            T, B, C_sub, H, N = x_seq.shape  # W,B,C/N,H,N
        device = x_seq.device
        dtype = x_seq.dtype
        
        if self.no_filter == False:

        
            (tau_matrix, tau_vec_init, tau_vec_rest, tau) = \
            self._build_tau_terms(T, device, dtype)

            if self.decay_input:
                
                if self.soma_dim == 3:
                    div_tau = tau.view(1, 1, 1, 1, 1, N)
                else:
                    div_tau = tau.view(1, 1, 1, 1, N)    
                x_seq = x_seq / div_tau

            y_branches = []
            for n in range(self.num_branches):
                x_n = x_seq[..., n].reshape(T, -1)
                y_n_flat = torch.matmul(tau_matrix[n], x_n)
                if self.soma_dim == 3:
                    y_n = y_n_flat.view(T, B, C_sub, H, W)
                else:
                    y_n = y_n_flat.view(T, B, C_sub, H)    
                
                # Init & Rest
                t_init = tau_vec_init[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_init[n].view(T, 1, 1, 1)
                t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_rest[n].view(T, 1, 1, 1)
                
                v_init = self.v
                if isinstance(v_init, torch.Tensor):
                    v_in = v_init[..., n].detach().unsqueeze(0)
                else:
                    v_in = v_init
                    
                y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
                y_branches.append(y_n)

            y = torch.stack(y_branches, dim=-1) # (T, B, ..., N)
            
        
        else:
            y = x_seq
              
        if self.store_v_seq:
            self.v_seq = y

        self.v = y[-1].detach()
        
        if self.gating:

            m_mod = self._compute_astro_modulation(y)
            y = y * m_mod
            if self.bn:
                
                
                if self.soma_dim == 3:
                        y_permuted = y.permute(0, 1, 5, 2, 3, 4).contiguous()
                        y_reshaped = y_permuted.view(T * B, N * C_sub, H, W)
                        
                        y_normed_reshaped = self.branch_bn(y_reshaped)
                        
                        y_normed = y_normed_reshaped.view(T, B, N, C_sub, H, W)

                        y_normed = y_normed.permute(0, 1, 3, 4, 5, 2).contiguous()

                else:
                        y_permuted = y.permute(0, 1, 4, 2, 3).contiguous()
                        
                        # 2. 融合维度：(Batch, Channels, Length)
                        # 目标形状: (T * B, C * N, H)
                        y_reshaped = y_permuted.view(T * B, N * C_sub, H)
                        
                        # 3. 过 BatchNorm1d
                        y_normed_reshaped = self.branch_bn(y_reshaped)
                        
                        # 4. 还原形状：先解开 view -> (T, B, C, N, H)
                        y_normed = y_normed_reshaped.view(T, B, N, C_sub, H)
                        
                        # 5. 再把 N 移回最后 -> (T, B, C, H, N)
                        y_normed = y_normed.permute(0, 1, 3, 4, 2).contiguous()        

                #m_mod = self._compute_astro_modulation(y_normed)
                y_out = y_normed
                if self.last_sigmoid == True:
                    gate = torch.sigmoid(self.gate_alpha * y_out + self.gate_beta)  ##这里用的y_out,而不是y？
                    y_out = y_out * gate
                if self.last_tanh == True:
                    gate = torch.tanh(self.gate_alpha * y_out + self.gate_beta)  ##这里用的y_out,而不是y？
                    y_out = y_out * gate   
                
            else:
                y_out = y
                if self.last_sigmoid == True:
                    gate_factor = torch.sigmoid(self.gate_scale * y_out + self.gate_beta2)
                    y_out = y_out * gate_factor
                if self.last_tanh == True:
                    gate_factor = torch.tanh(self.gate_scale * y_out + self.gate_beta2)
                    y_out = y_out * gate_factor        
        else:
            y_out = y


        if self.res == True:
            y_out = y_out + self.skip_weight * x_seq
          
               
            
        return y_out



class AdvancedNGCUDendCompartment(BaseDendCompartment):
    """
    Advanced Neuron-Glia Coupling Unit (NGCU) with Multi-Scale Dendrites.
    
    Features:
    1. Microdomain-level Glia Modulation (Independent C_t per branch).
    2. Polarity-specific gating (Excitation vs Inhibition dual thresholds).
    3. Multi-scale passive dendritic integration (Matrix accelerated).
    4. Tripartite synapse emulation (Multiplicative Gain + Additive Bias).
    """

    def __init__(
        self,
        num_branches,
        c_sub=1,
        init_tau=[1.2, 5.0],
        #init_tau=[1.2, 8.0],
        soma_dim=3,
        decay_input: bool = True,
        bn=True,
        res=False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
        use_astro = True,
        last_sigmoid = True,
        last_tanh = False
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        self.soma_dim = soma_dim
        self.bn = bn
        self.res = res
        self.no_filter = no_filter
        self.last_sigmoid = last_sigmoid
        self.last_tanh = last_tanh
        self.use_astro = use_astro
        self.c_sub = c_sub
        self.num_branches = num_branches
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.skip_weight = nn.Parameter(torch.tensor(0.01))

        # ---------------- 树突局部积分参数 ----------------
        if init_tau is None:
            tau_data = torch.linspace(2.0, 10.0, steps=num_branches)
        elif isinstance(init_tau, (float, int)):
            tau_data = torch.full((num_branches,), float(init_tau))
        else:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
        self.tau_branches = nn.Parameter(torch.tensor(tau_data, dtype=torch.float32))    

        if self.bn:
            self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) if self.soma_dim == 3 else nn.BatchNorm1d(self.c_sub, affine=True, momentum=0.1)  
            nn.init.constant_(self.branch_bn.weight, 1.0)
            nn.init.constant_(self.branch_bn.bias, 0.0)
            if last_sigmoid == True or last_tanh == True:
                if self.soma_dim == 3:
                        # BN + Gating 模式 (静态图/DVS通用)
                    self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                    self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches))
                else:
                    self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, num_branches))
                    self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, num_branches))
        else:
            if last_sigmoid == True or last_tanh == True:   
                self.gate_scale = nn.Parameter(torch.tensor(1.0))
                self.gate_beta2 = nn.Parameter(torch.tensor(0.0))        
        # ---------------- NGCU 胶质细胞学习参数 ----------------
        
        # 1. 极性写入通道 (Upward Writing)
        self.theta_exc = nn.Parameter(torch.tensor(0.0),requires_grad=False) ##
        self.theta_inh = nn.Parameter(torch.tensor(0.0),requires_grad=False) ##
        self.theta = nn.Parameter(torch.full((num_branches,), 0.5)) ##
        #self.theta = nn.Parameter(torch.tensor((0.7, 0.3)))
        self.w_exc = nn.Parameter(torch.tensor(1.0))
        self.w_inh = nn.Parameter(torch.tensor(1.0))
        
        self.lambda_local = nn.Parameter(torch.tensor(0.4)) ## 适当调大

        # 4. 下行调制生成 (Downward Modulation)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.w_b = nn.Parameter(torch.tensor(0.1))

    def _build_tau_terms(self, T: int, device, dtype):
        # ... (与你原本的矩阵构建代码完全一致，略去以节省空间，直接复用你原有代码) ...
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-5)
        a = 1.0 - 1.0 / tau
        t = torch.arange(T, device=device, dtype=dtype)
        i = t[:, None]
        j = t[None, :]
        diff = i - j
        mask = diff >= 0
        tau_matrix = torch.zeros((len(tau), T, T), device=device, dtype=dtype)
        for n in range(len(tau)):
            tau_matrix[n] = tau_matrix[n].masked_scatter(mask, (a[n] ** diff[mask]).to(dtype))
        tau_vec_init = a.unsqueeze(1) ** (t + 1.0)
        tau_vec_rest = 1.0 - tau_vec_init
        return tau_matrix, tau_vec_init, tau_vec_rest, tau

    def _compute_ngcu_modulation(self, y_seq):
        """
        核心物理引擎：计算胶质微结构域扩散、蓝斑核唤醒、以及向心全局记忆积分。
        """
        T = y_seq.shape[0]
        B = y_seq.shape[1]
        N = self.num_branches
        device = y_seq.device
        
        # 初始化胶质状态
        C_local = torch.zeros((B, N), device=device)   # 微结构域状态 (各分支独立)
        #C_astro = torch.zeros((B, 1), device=device)   # 胶质胞体宏观状态 (全局唯一)

        G_seq, B_seq = [], []
        C_seq_list = []

        # 预计算 Sigmoid 约束的衰减率，确保其物理意义 (介于0到1之间)
        lam_l = self.lambda_local

        for t in range(T):
            # 1. 特征降维聚合 (注意这里去掉了 abs()，保留正负极性！)
            E_raw = F.relu(y_seq[t] - self.theta_exc)  ##
            I_raw = F.relu(-y_seq[t] - self.theta_inh) ##

            # 2. 然后再进行降维聚合
            if self.soma_dim == 3:
                E_t = E_raw.mean(dim=(1, 2, 3)) # (B, N)
                I_t = I_raw.mean(dim=(1, 2, 3)) # (B, N)
            else:
                E_t = E_raw.mean(dim=(1, 2))
                I_t = I_raw.mean(dim=(1, 2))

            # 3. 计算有饱和约束的写入量
            #Psi_t = torch.tanh(self.w_exc * E_t - self.w_inh * I_t)
            Psi_t = self.w_exc * E_t - self.w_inh * I_t  
            # 局部状态演化
            C_local = (1 - lam_l) * C_local + lam_l * torch.tanh(F.relu(Psi_t - self.theta))  ####
            #C_local = (1 - lam_l) * C_local + lam_l * F.relu(Psi_t - self.theta)
            C_seq_list.append(C_local)
            # 5. 生成下行调制 (局部乘性增益 + 全局/局部混合偏置)
            #G_t = 1.0 + self.alpha * torch.tanh(self.beta * C_local) # (B, N)
            G_t = 1.0 + self.alpha * C_local
            #B_t = torch.tanh(self.w_b * C_local) # (B, N)
            B_t = self.w_b * C_local

            G_seq.append(G_t)
            B_seq.append(B_t)


        # 堆叠回 T 维度并调整形状以准备广播相乘
        G_seq = torch.stack(G_seq, dim=0) # (T, B, N)
        B_seq = torch.stack(B_seq, dim=0) # (T, B, N)
        C_seq = torch.stack(C_seq_list, dim=0)
        self.current_C_seq = C_seq

        if self.soma_dim == 3:
            G_seq = G_seq.view(T, B, 1, 1, 1, N)
            B_seq = B_seq.view(T, B, 1, 1, 1, N)
        else:
            G_seq = G_seq.view(T, B, 1, 1, N)
            B_seq = B_seq.view(T, B, 1, 1, N)

        return G_seq, B_seq
    
    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-3)

        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        if self.decay_input:
            v = self.single_step_decay_input(self.v, x, self.v_rest, tau)
        else:
            v = self.single_step_not_decay_input(self.v, x, self.v_rest, tau)

        # IMPORTANT:
        # detach state to avoid cross-iteration graph leakage
        self.v = v.detach()
        return v

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.soma_dim == 3:
            T, B, C_sub, H, W, N = x_seq.shape
        else:
            T, B, C_sub, H, N = x_seq.shape
            
        device, dtype = x_seq.device, x_seq.dtype
        
        if not self.no_filter:
            tau_matrix, tau_vec_init, tau_vec_rest, tau = self._build_tau_terms(T, device, dtype)

            if self.decay_input:
                div_tau = tau.view(1, 1, 1, 1, 1, N) if self.soma_dim == 3 else tau.view(1, 1, 1, 1, N)    
                x_seq = x_seq / div_tau

            y_branches = []
            for n in range(self.num_branches):
                x_n = x_seq[..., n].reshape(T, -1)
                y_n_flat = torch.matmul(tau_matrix[n], x_n)
                
                y_n = y_n_flat.view(T, B, C_sub, H, W) if self.soma_dim == 3 else y_n_flat.view(T, B, C_sub, H)    
                
                t_init = tau_vec_init[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_init[n].view(T, 1, 1, 1)
                t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_rest[n].view(T, 1, 1, 1)
                
                v_in = self.v[..., n].detach().unsqueeze(0) if isinstance(self.v, torch.Tensor) else self.v
                y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
                y_branches.append(y_n)

            y = torch.stack(y_branches, dim=-1) # (T, B, ..., N)
        else:
            y = x_seq
              
        if self.store_v_seq:
            self.v_seq = y
        self.v = y[-1].detach()
        if self.use_astro:
            G_mod, B_mod = self._compute_ngcu_modulation(y)
            # 计算高级 NGCU 调制 (包含所有时空动力学)
            y = G_mod * y + B_mod
        if self.bn:
            if self.soma_dim == 3:
                y_permuted = y.permute(0, 1, 5, 2, 3, 4).contiguous()
                y_reshaped = y_permuted.view(T * B, N * C_sub, H, W)
                y_normed = self.branch_bn(y_reshaped).view(T, B, N, C_sub, H, W)
                y_normed = y_normed.permute(0, 1, 3, 4, 5, 2).contiguous()
            else:
                y_permuted = y.permute(0, 1, 4, 2, 3).contiguous()
                y_reshaped = y_permuted.view(T * B, N * C_sub, H)
                y_normed = self.branch_bn(y_reshaped).view(T, B, N, C_sub, H)
                y_normed = y_normed.permute(0, 1, 3, 4, 2).contiguous()        

            # 应用三方突触调制: y_out = G * y_bn + B
            y_out = y_normed
            #y_out = y_normed
            if self.last_sigmoid == True:
                gate = torch.sigmoid(self.gate_alpha * y_out + self.gate_beta)
                y_out = y_out * gate
            if self.last_tanh == True:
                gate = torch.tanh(self.gate_alpha * y_out + self.gate_beta)
                y_out = y_out * gate    
        else:
            #y_out = G_mod * y + B_mod
            y_out = y
            if self.last_sigmoid == True:
                gate_factor = torch.sigmoid(self.gate_scale * y_out + self.gate_beta2)
                y_out = y_out * gate_factor
            if self.last_tanh == True:
                gate_factor = torch.tanh(self.gate_scale * y_out + self.gate_beta2)
                y_out = y_out * gate_factor   

        if self.res:
            y_out = y_out + self.skip_weight * x_seq
            
        return y_out






class PAComponentDendCompartment(BaseDendCompartment):
    """Dendritic compartment with passive and active voltage components.

    The passive component acts just like a leaky integrator without a firing
    mechanism, while the active voltage component is a function of the passive 
    voltage component. The overall voltage is the sum of active and passive 
    components: 
        v[t] = va[t] + vp[t] = f_dca(vp[t]) + vp[t]
    Get inspiration from:
    Legenstein, R., & Maass, W. (2011). Branch-specific plasticity enables 
    self-organization of nonlinear computation in single neurons. The Journal 
    of Neuroscience: The Official Journal of the Society for Neuroscience, 
    31(30), 10787–10802. https://doi.org/10.1523/JNEUROSCI.5684-10.2011


    Attributes:
        v (Union[float, torch.Tensor]): voltage of the dendritic compartment(s)
            at the current time step.
        va (Union[float, torch.Tensor]): the active component of the 
            compartmental voltage at the current time step.
        vp (Union[float, torch.Tensor]): the passive component of the 
            compartmental voltage at the current time step.
        step_mode (str): "s" for single-step mode, and "m" for multi-step mode.
        store_v_seq (bool): whether to store the compartmental potential at 
            every time step when using multi-step mode. If True, there is 
            another attribute called v_seq.
        store_vp_seq (bool): whether to store the passive component of the 
            compartmental potential at every time step when using multi-step 
            mode. If True, there is another attribute called vp_seq.
        store_va_seq (bool): whether to store the active component of the 
            compartmental potential at every time step when using multi-step 
            mode. If True, there is another attribute called va_seq.
        tau(float): the time constant for the passive component.
        decay_input (bool, optional): whether the input to the compartments
            should be divided by tau.
        v_rest (float, optional): resting potential.
        f_dca (Callable): the dendritic compartment activation function, mapping
            the passive voltage component to the active component. The input and
            output should have the same shape.
    """

    def __init__(
        self, tau: float = 2., decay_input: bool = True, v_rest: float = 0., 
        f_dca: Callable = lambda x: 0., step_mode: str = "s", 
        store_v_seq: bool = False, store_vp_seq: bool = False, 
        store_va_seq: bool = False
    ):
        """The constructor of PAComponentDendCompartment

        Args:
            tau (float, optional): the time constant. Defaults to 2.
            decay_input (bool, optional): whether the input to the compartments
                should be divided by tau. Defaults to True.
            v_rest (float, optional): resting potential. Defaults to 0..
            f_dc (Callable): the dendritic compartment activation function, 
                mapping the passive voltage component to the active component. 
                The input and output should have the same shape. Defaults to 
                the constant zero.
            step_mode (str, optional): "s" for single-step mode, and "m" for
                multi-step mode. Defaults to "s".
            store_v_seq (bool, optional): whether to store the compartmental 
                potential at every time step when using multi-step mode. 
                Defaults to False.
            store_vp_seq (bool, optional): whether to store the passive 
                component of the compartmental potential at every time step when 
                using multi-step mode. Defaults to False.
            store_v_seq (bool, optional): whether to store the active component 
                of the compartmental potential at every time step when using 
                multi-step mode. Defaults to False.
        """
        super().__init__(v_rest, step_mode, store_v_seq)
        self.tau = tau
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.f_dca = f_dca
        self.register_memory("va", 0.)
        self.register_memory("vp", v_rest)
        self.store_vp_seq = store_vp_seq
        self.store_va_seq = store_va_seq

    @property
    def store_vp_seq(self) -> bool:
        return self._store_vp_seq

    @store_vp_seq.setter
    def store_vp_seq(self, val: bool):
        self._store_vp_seq = val
        if val and (not hasattr(self, "vp_seq")):
            self.register_memory("vp_seq", None)

    @property
    def store_va_seq(self) -> bool:
        return self._store_va_seq

    @store_va_seq.setter
    def store_va_seq(self, val: bool):
        self._store_va_seq = val
        if val and (not hasattr(self, "va_seq")):
            self.register_memory("va_seq", None)

    def v_float2tensor(self, x: torch.Tensor):
        """If self.v | vp | va is a float, turn it into a tensor with x's shape.
        """
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)
        if isinstance(self.va, float):
            v_init = self.va
            self.va = torch.full_like(x.data, v_init)
        if isinstance(self.vp, float):
            v_init = self.vp
            self.vp = torch.full_like(x.data, v_init)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.decay_input:
            self.vp = self.vp + (x - (self.vp - self.v_rest)) / self.tau
        else:
            self.vp = self.vp + x - (self.vp - self.v_rest) / self.tau

        self.va = self.f_dca(self.vp)
        self.v = self.vp + self.va
        return self.v

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        if self.store_vp_seq:
            vp_seq = []
        if self.store_va_seq:
            va_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
            if self.store_vp_seq:
                vp_seq.append(self.vp)
            if self.store_va_seq:
                va_seq.append(self.va)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_vp_seq:
            self.vp_seq = torch.stack(vp_seq)
        if self.store_va_seq:
            self.va_seq = torch.stack(va_seq)
        return torch.stack(y_seq)
