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
        
        self.tau_branches = nn.Parameter(tau_data.clone().detach().float())
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

        self.tau_branches = nn.Parameter(tau_data.clone().detach().float())    
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
        #init_tau=[1.2, 1.5, 4.0, 6.0],
        init_tau=[1.5, 5.0],
        soma_dim=3,
        decay_input: bool = True,
        bn=True,  ##
        bn_old=False,
        res=False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
        use_astro = True,
        last_sigmoid = True,
        last_tanh = False,
        use_event_write = False,
        event_threshold: float = 0.3,
        event_slope: float = 8.0,
        event_delta_weight: float = 0.5,
        spatial_astro: bool = False,
        astro_pool_kernel: int = 3,
        astro_active_eps: float = 1e-6,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        self.soma_dim = soma_dim
        self.bn = bn
        self.bn_old = bn_old
        self.res = res
        self.no_filter = no_filter
        self.last_sigmoid = last_sigmoid
        self.last_tanh = last_tanh
        self.use_astro = use_astro
        self.use_event_write = use_event_write
        self.spatial_astro = spatial_astro
        self.astro_pool_kernel = int(astro_pool_kernel)
        self.astro_active_eps = float(astro_active_eps)
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
        self.tau_branches = nn.Parameter(tau_data.clone().detach().float())   

        if self.bn or self.bn_old:
            if self.bn and not self.bn_old:
                self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) if self.soma_dim == 3 else nn.BatchNorm1d(self.c_sub, affine=True, momentum=0.1)
            if self.bn_old and not self.bn:
                self.branch_bn = nn.BatchNorm1d(num_branches, affine=True, momentum=0.1)      
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
        self.theta = nn.Parameter(torch.full((num_branches,), 0.2),requires_grad=True) ##
        #self.theta = nn.Parameter(torch.tensor((0.7, 0.3)))
        self.w_exc = nn.Parameter(torch.tensor(1.0))
        self.w_inh = nn.Parameter(torch.tensor(1.0))
        
        self.lambda_local = nn.Parameter(torch.tensor(-1.0)) ## 适当调大
        self.event_threshold = nn.Parameter(torch.tensor(event_threshold, dtype=torch.float32))
        self.event_delta_weight = nn.Parameter(torch.tensor(event_delta_weight, dtype=torch.float32))
        self.event_slope = float(event_slope)

        # 4. 下行调制生成 (Downward Modulation)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.w_b = nn.Parameter(torch.tensor(0.1))

    def _build_tau_terms(self, T: int, device, dtype):
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

    def _local_active_mean(self, x: torch.Tensor):
        """Channel-collapsed local active mean for astro microdomains."""
        kernel = self.astro_pool_kernel
        energy = x.mean(dim=1, keepdim=True)
        if kernel <= 1:
            return energy

        mask = (energy > 0).to(energy.dtype)
        padding = kernel // 2
        if self.soma_dim == 3:
            B, _, H, W, N = energy.shape
            energy_flat = energy.permute(0, 4, 1, 2, 3).reshape(B * N, 1, H, W)
            mask_flat = mask.permute(0, 4, 1, 2, 3).reshape(B * N, 1, H, W)
            numerator = F.avg_pool2d(
                energy_flat * mask_flat,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )
            denominator = F.avg_pool2d(
                mask_flat,
                kernel_size=kernel,
                stride=1,
                padding=padding,
                count_include_pad=False,
            ).clamp_min(self.astro_active_eps)
            pooled = numerator / denominator
            return pooled.view(B, N, 1, H, W).permute(0, 2, 3, 4, 1).contiguous()

        B, _, H, N = energy.shape
        energy_flat = energy.permute(0, 3, 1, 2).reshape(B * N, 1, H)
        mask_flat = mask.permute(0, 3, 1, 2).reshape(B * N, 1, H)
        numerator = F.avg_pool1d(
            energy_flat * mask_flat,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        denominator = F.avg_pool1d(
            mask_flat,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        ).clamp_min(self.astro_active_eps)
        pooled = numerator / denominator
        return pooled.view(B, N, 1, H).permute(0, 2, 3, 1).contiguous()

    def _compute_ngcu_modulation(self, y_seq):
        """
        核心物理引擎：计算胶质微结构域扩散、蓝斑核唤醒、以及向心全局记忆积分。
        """
        T = y_seq.shape[0]
        B = y_seq.shape[1]
        N = self.num_branches
        device = y_seq.device
        
        # 初始化胶质状态
        if self.spatial_astro:
            if self.soma_dim == 3:
                H, W = y_seq.shape[3], y_seq.shape[4]
                C_local = torch.zeros((B, 1, H, W, N), device=device, dtype=y_seq.dtype)
            else:
                H = y_seq.shape[3]
                C_local = torch.zeros((B, 1, H, N), device=device, dtype=y_seq.dtype)
        else:
            C_local = torch.zeros((B, N), device=device, dtype=y_seq.dtype)   # 微结构域状态 (各分支独立)
        #C_astro = torch.zeros((B, 1), device=device)   # 胶质胞体宏观状态 (全局唯一)

        G_seq, B_seq = [], []
        C_seq_list = []
        event_gate_seq = []

        # 预计算 Sigmoid 约束的衰减率，确保其物理意义 (介于0到1之间)
        lam_l = torch.sigmoid(self.lambda_local)

        for t in range(T):
            #energy_seq = y_seq[t].abs().mean(dim=(1, 2, 3)) if self.soma_dim == 3 else y_seq[t].abs().mean(dim=(1, 2))
            #E_raw = F.relu(y_seq[t] - self.theta_exc)  ##
            #I_raw = F.relu(-y_seq[t] - self.theta_inh) ##

            # 2. 然后再进行降维聚合
            if self.spatial_astro:
                #E_t = self._local_active_mean(E_raw)
                #I_t = self._local_active_mean(I_raw)
                energy = self._local_active_mean(energy_seq)  #  这里要完全重写或者放弃!!!
                theta = self.theta.view(1, 1, 1, 1, N) if self.soma_dim == 3 else self.theta.view(1, 1, 1, N)
            else:
                if self.soma_dim == 3:
                    #E_t = E_raw.mean(dim=(1, 2, 3)) # (B, N)
                    #I_t = I_raw.mean(dim=(1, 2, 3)) # (B, N)
                    energy_seq = y_seq[t].abs().mean(dim=(1, 2, 3))
                else:
                    #E_t = E_raw.mean(dim=(1, 2))
                    #I_t = I_raw.mean(dim=(1, 2))
                    energy_seq = y_seq[t].abs().mean(dim=(1, 2))
                theta = self.theta

            # 3. 计算有饱和约束的写入量
            #Psi_t = torch.tanh(self.w_exc * E_t - self.w_inh * I_t)
            #Psi_t = self.w_exc * E_t + self.w_inh * I_t
            Psi_t = energy_seq  
            # 局部状态演化
            write_t = torch.tanh(F.relu(Psi_t - theta))
            if self.use_event_write:
                branch_activity = Psi_t
                delta_drive = (write_t - C_local).abs()
                event_drive = branch_activity + self.event_delta_weight.abs() * delta_drive
                event_gate = torch.sigmoid(self.event_slope * (event_drive - self.event_threshold.abs()))
                #event_gate = event_gate * (event_drive > 0).to(event_gate.dtype)
            else:
                event_gate = torch.ones_like(write_t)
            C_local = (1 - lam_l * event_gate) * C_local + lam_l * event_gate * write_t  ####
            #C_local = (1 - lam_l) * C_local + lam_l * F.relu(Psi_t - self.theta)
            C_seq_list.append(C_local)
            event_gate_seq.append(event_gate.detach())
            # 5. 生成下行调制 (局部乘性增益 + 全局/局部混合偏置)
            G_t = 1.0 + self.alpha * torch.tanh(self.beta * C_local) # (B, N)
            #G_t = 1.0 + self.alpha * C_local
            B_t = torch.tanh(self.w_b * C_local) # (B, N)
            #B_t = self.w_b * C_local

            G_seq.append(G_t)
            B_seq.append(B_t)


        # 堆叠回 T 维度并调整形状以准备广播相乘
        G_seq = torch.stack(G_seq, dim=0)
        B_seq = torch.stack(B_seq, dim=0)
        C_seq = torch.stack(C_seq_list, dim=0)
        self.current_C_seq = C_seq
        self.current_event_gate_seq = torch.stack(event_gate_seq, dim=0)

        if self.spatial_astro:
            return G_seq, B_seq

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
            #y = G_mod * y + B_mod
        if self.bn or self.bn_old:    
            if self.bn and not self.bn_old:
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

            if self.bn_old and not self.bn:
                
                y_normed = self.branch_bn(y.view(-1,N))                
                y_normed = y_normed.reshape(y.shape)
                # 应用三方突触调制: y_out = G * y_bn + B 
            
            y_out = y_normed
            if self.use_astro:
               
                y_out = G_mod * y_out + B_mod
                #y_out = y_normed
            if self.last_sigmoid == True:
                gate = torch.sigmoid(self.gate_alpha * y_out + self.gate_beta)
                y_out = y_out * gate
            if self.last_tanh == True:
                gate = torch.tanh(self.gate_alpha * y_out + self.gate_beta)
                y_out = y_out * gate    
        else:
            if self.use_astro:
                y_out = G_mod * y + B_mod
            else:    
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


class HierarchicalTrunkDistalDendCompartment(BaseDendCompartment):
    """Pure dendritic hierarchy with B branches and K compartments per branch.

    The input compartment dimension is interpreted as branch-major groups:
    ``[b0_k0, b0_k1, ..., b1_k0, b1_k1, ...]``.  In every branch, ``k0`` is the
    proximal trunk that carries the main signal, while the remaining distal
    compartments modulate that trunk through a gated nonlinear interaction.

    This module deliberately contains no astrocyte state.  It is meant to
    replace the old "N parallel integrators" dendrite with a local dendritic
    computation before the soma/astro loop is applied elsewhere.
    """

    def __init__(
        self,
        num_branches,
        compartments_per_branch=2,
        c_sub=1,
        init_tau=[1.5, 2.0, 5.0, 6.0], #
        soma_dim=3,
        decay_input: bool = True,
        bn=True, #
        res=False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
        last_sigmoid=True, #
        last_tanh=False,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")  ##不一定要有2个以上compartment，至少要有一个trunk和一个distal，但也可以只有trunk没有distal，此时就退化成纯积分器了

        self.soma_dim = soma_dim
        self.bn = bn
        self.res = res
        self.no_filter = no_filter
        self.last_sigmoid = last_sigmoid
        self.last_tanh = last_tanh
        self.c_sub = c_sub
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.num_compartments = self.num_branches * self.compartments_per_branch
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.skip_weight = nn.Parameter(torch.tensor(0.01), requires_grad=False)

        tau_data = self._make_tau_data(init_tau)
        self.tau_compartments = nn.Parameter(tau_data.clone().detach().float())

        self.distal_gate_weight = nn.Parameter(torch.ones(self.num_branches, self.distal_count))
        self.trunk_gate_weight = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_gate_threshold = nn.Parameter(torch.full((self.num_branches, self.distal_count), 0.5))
        self.distal_response_weight = nn.Parameter(torch.ones(self.num_branches, self.distal_count))
        self.distal_mix_logits = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.branch_gain = nn.Parameter(torch.full((self.num_branches,), 0.1))
        self.branch_bias = nn.Parameter(torch.zeros(self.num_branches))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))

        if self.bn:
            if self.c_sub % self.compartments_per_branch != 0:
                raise ValueError(
                    "c_sub should be the channel count before dendritic reshape "
                    "and must be divisible by compartments_per_branch when bn=True"
                )
            bn_channels = self.c_sub // self.compartments_per_branch
            self.branch_bn = (
                nn.BatchNorm2d(bn_channels, affine=True, momentum=0.1)
                if self.soma_dim == 3
                else nn.BatchNorm1d(bn_channels, affine=True, momentum=0.1)
            )
            nn.init.constant_(self.branch_bn.weight, 1.0)
            nn.init.constant_(self.branch_bn.bias, 0.0)

        if self.last_sigmoid or self.last_tanh:
            self.gate_alpha = nn.Parameter(torch.ones(self.num_branches))
            self.gate_beta = nn.Parameter(torch.zeros(self.num_branches))

        self.branch_mod_seq = None
        self.distal_gate_seq = None
        self.distal_response_seq = None

    def _make_tau_data(self, init_tau):
        if init_tau is None:
            per_branch_tau = torch.linspace(1.2, 6.0, steps=self.compartments_per_branch)
            return per_branch_tau.repeat(self.num_branches)
        if isinstance(init_tau, (float, int)):
            return torch.full((self.num_compartments,), float(init_tau))

        tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
        if tau_data.numel() == 2:
            per_branch_tau = torch.linspace(
                float(tau_data.flatten()[0]),
                float(tau_data.flatten()[-1]),
                steps=self.compartments_per_branch,
            )
            return per_branch_tau.repeat(self.num_branches)
        if tau_data.numel() == self.compartments_per_branch:
            return tau_data.flatten().repeat(self.num_branches)
        if tau_data.numel() == self.num_compartments:
            return tau_data.reshape(-1)
        raise ValueError(
            "init_tau must be scalar, length 2, length compartments_per_branch, "
            "or match num_branches * compartments_per_branch"
        )

    def _build_tau_terms(self, T: int, device, dtype):
        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-5)
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

    def _assert_input_shape(self, x: torch.Tensor):
        if x.shape[-1] != self.num_compartments:
            raise ValueError(
                f"Expected last input dimension to be num_branches * "
                f"compartments_per_branch = {self.num_compartments}, "
                f"but got {x.shape[-1]}"
            )

    def _view_distal_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 2)), self.num_branches, self.distal_count)

    def _view_branch_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 1)), self.num_branches)

    def _reshape_to_branches(self, x: torch.Tensor):
        self._assert_input_shape(x)
        return x.reshape(*x.shape[:-1], self.num_branches, self.compartments_per_branch)

    def _compute_branch_output(self, compartment_state: torch.Tensor, raw_input: torch.Tensor = None):
        branch_state = self._reshape_to_branches(compartment_state)
        trunk = branch_state[..., 0]
        distal = branch_state[..., 1:]

        gate = torch.sigmoid(
            self._view_distal_param(self.distal_gate_weight, distal) * distal
            + self._view_distal_param(self.trunk_gate_weight, distal) * trunk.unsqueeze(-1)
            - self._view_distal_param(self.distal_gate_threshold, distal)
        )
        response = torch.tanh(self._view_distal_param(self.distal_response_weight, distal) * distal)
        mix = torch.softmax(self.distal_mix_logits, dim=-1)
        branch_mod = (self._view_distal_param(mix, distal) * gate * response).sum(dim=-1)

        self.branch_mod_seq = branch_mod
        self.distal_gate_seq = gate
        self.distal_response_seq = response

        branch_mod = torch.tanh(branch_mod)
        gain = 1.0 + self._view_branch_param(self.branch_gain, trunk) * branch_mod
        bias = self._view_branch_param(self.branch_bias, trunk) * branch_mod
        y = self._view_branch_param(self.branch_strength, trunk) * (trunk * gain + bias)

        if self.res:
            if raw_input is None:
                trunk_input = trunk
            else:
                trunk_input = self._reshape_to_branches(raw_input)[..., 0]
            y = y + self.skip_weight * trunk_input
        return y

    def _apply_branch_bn(self, y: torch.Tensor):
        if not self.bn:
            return y

        if self.soma_dim == 3:
            if y.dim() == 6:
                T, batch_size, C_sub, H, W, N = y.shape
                flat_channels = N * C_sub
                self._check_bn_channels(flat_channels)
                y_permuted = y.permute(0, 1, 5, 2, 3, 4).contiguous()
                y_reshaped = y_permuted.view(T * batch_size, flat_channels, H, W)
                y_normed = self.branch_bn(y_reshaped).view(T, batch_size, N, C_sub, H, W)
                return y_normed.permute(0, 1, 3, 4, 5, 2).contiguous()
            if y.dim() == 5:
                batch_size, C_sub, H, W, N = y.shape
                flat_channels = N * C_sub
                self._check_bn_channels(flat_channels)
                y_permuted = y.permute(0, 4, 1, 2, 3).contiguous()
                y_reshaped = y_permuted.view(batch_size, flat_channels, H, W)
                y_normed = self.branch_bn(y_reshaped).view(batch_size, N, C_sub, H, W)
                return y_normed.permute(0, 2, 3, 4, 1).contiguous()
        else:
            if y.dim() == 5:
                T, batch_size, C_sub, H, N = y.shape
                flat_channels = N * C_sub
                self._check_bn_channels(flat_channels)
                y_permuted = y.permute(0, 1, 4, 2, 3).contiguous()
                y_reshaped = y_permuted.view(T * batch_size, flat_channels, H)
                y_normed = self.branch_bn(y_reshaped).view(T, batch_size, N, C_sub, H)
                return y_normed.permute(0, 1, 3, 4, 2).contiguous()
            if y.dim() == 4:
                batch_size, C_sub, H, N = y.shape
                flat_channels = N * C_sub
                self._check_bn_channels(flat_channels)
                y_permuted = y.permute(0, 3, 1, 2).contiguous()
                y_reshaped = y_permuted.view(batch_size, flat_channels, H)
                y_normed = self.branch_bn(y_reshaped).view(batch_size, N, C_sub, H)
                return y_normed.permute(0, 2, 3, 1).contiguous()

        raise ValueError(f"Unexpected branch output shape for soma_dim={self.soma_dim}: {tuple(y.shape)}")

    def _check_bn_channels(self, flat_channels: int):
        if self.branch_bn.num_features != flat_channels:
            raise RuntimeError(
                f"branch_bn was initialized with {self.branch_bn.num_features} channels, "
                f"but branch output has {flat_channels}. When using this compartment, "
                f"pass c_sub as the pre-reshape channel count: C_out * "
                f"(num_branches * compartments_per_branch)."
            )

    def _view_gate_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 1)), self.num_branches)

    def _apply_output_gate(self, y: torch.Tensor):
        if self.last_sigmoid:
            gate = torch.sigmoid(self._view_gate_param(self.gate_alpha, y) * y + self._view_gate_param(self.gate_beta, y))
            y = y * gate
        if self.last_tanh:
            gate = torch.tanh(self._view_gate_param(self.gate_alpha, y) * y + self._view_gate_param(self.gate_beta, y))
            y = y * gate
        return y

    def single_step_forward(self, x: torch.Tensor):
        self._assert_input_shape(x)
        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        tau = tau.view(*([1] * (x.dim() - 1)), self.num_compartments)
        if self.no_filter:
            v = x
        elif self.decay_input:
            v = self.v + (x - (self.v - self.v_rest)) / tau
        else:
            v = self.v - (self.v - self.v_rest) / tau + x

        self.v = v.detach()
        y = self._compute_branch_output(v, x)
        y = self._apply_branch_bn(y)
        y = self._apply_output_gate(y)
        return y

    def multi_step_forward(self, x_seq: torch.Tensor):
        self._assert_input_shape(x_seq)
        if self.soma_dim == 3:
            T, batch_size, C_sub, H, W, N = x_seq.shape
        else:
            T, batch_size, C_sub, H, N = x_seq.shape

        device, dtype = x_seq.device, x_seq.dtype
        dend_input = x_seq

        if not self.no_filter:
            tau_matrix, tau_vec_init, tau_vec_rest, tau = self._build_tau_terms(T, device, dtype)
            tau_shape = [1] * x_seq.dim()
            tau_shape[-1] = N
            if self.decay_input:
                dend_input = dend_input / tau.view(*tau_shape)

            y_compartments = []
            for n in range(self.num_compartments):
                x_n = dend_input[..., n].reshape(T, -1)
                y_n_flat = torch.matmul(tau_matrix[n], x_n)
                y_n = (
                    y_n_flat.view(T, batch_size, C_sub, H, W)
                    if self.soma_dim == 3
                    else y_n_flat.view(T, batch_size, C_sub, H)
                )

                t_init = (
                    tau_vec_init[n].view(T, 1, 1, 1, 1)
                    if self.soma_dim == 3
                    else tau_vec_init[n].view(T, 1, 1, 1)
                )
                t_rest = (
                    tau_vec_rest[n].view(T, 1, 1, 1, 1)
                    if self.soma_dim == 3
                    else tau_vec_rest[n].view(T, 1, 1, 1)
                )
                v_in = self.v[..., n].detach().unsqueeze(0) if isinstance(self.v, torch.Tensor) else self.v
                y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
                y_compartments.append(y_n)

            y_compartment = torch.stack(y_compartments, dim=-1)
        else:
            y_compartment = dend_input

        if self.store_v_seq:
            self.v_seq = y_compartment
        self.v = y_compartment[-1].detach()

        y = self._compute_branch_output(y_compartment, x_seq)
        y = self._apply_branch_bn(y)
        y = self._apply_output_gate(y)
        return y



class PureMultiScaleDendCompartment(BaseDendCompartment):
    """Multi-branch dendritic integrator without astrocyte modulation.

    This class keeps the dendritic part of :class:`AdvancedNGCUDendCompartment`:
    independent branch time constants, causal passive integration, optional
    branch normalization, and optional branch output gating. It intentionally
    does not maintain a glial state and does not generate gain/bias modulation.
    """

    def __init__(
        self,
        num_branches,
        c_sub=1,
        #init_tau=[1.5, 5.0],
        init_tau=[1.5, 2.0, 5.0, 6.0],
        soma_dim=3,
        decay_input: bool = True, 
        bn=True,  ##
        res=False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
        last_sigmoid=True,
        last_tanh=False,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        self.soma_dim = soma_dim
        self.bn = bn
        self.res = res
        self.no_filter = no_filter
        self.last_sigmoid = last_sigmoid
        self.last_tanh = last_tanh
        self.c_sub = c_sub
        self.num_branches = num_branches
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.skip_weight = nn.Parameter(torch.tensor(0.05),requires_grad=True)

        if init_tau is None:
            tau_data = torch.linspace(2.0, 10.0, steps=num_branches)
        elif isinstance(init_tau, (float, int)):
            tau_data = torch.full((num_branches,), float(init_tau))
        else:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
        if tau_data.numel() != num_branches:
            if tau_data.numel() == 2:
                tau_data = torch.linspace(float(tau_data[0]), float(tau_data[-1]), steps=num_branches)
            else:
                raise ValueError("init_tau must be scalar, length 2, or match num_branches")
        self.tau_branches = nn.Parameter(tau_data.clone().detach().float())

        if self.bn:
            self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) if self.soma_dim == 3 else nn.BatchNorm1d(self.c_sub, affine=True, momentum=0.1)
            nn.init.constant_(self.branch_bn.weight, 1.0)
            nn.init.constant_(self.branch_bn.bias, 0.0)
            if last_sigmoid or last_tanh:
                if self.soma_dim == 3:
                    self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                    self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches))
                else:
                    self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, num_branches))
                    self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, num_branches))
        else:
            if last_sigmoid or last_tanh:
                self.gate_scale = nn.Parameter(torch.tensor(1.0))
                self.gate_beta2 = nn.Parameter(torch.tensor(0.0))

    def _build_tau_terms(self, T: int, device, dtype):
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

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau_branches, min=1.0 + 1e-3)
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        view_shape = [1] * x.dim()
        view_shape[-1] = self.num_branches
        tau = tau.view(*view_shape)
        if self.decay_input:
            v = self.v + (x - (self.v - self.v_rest)) / tau
        else:
            v = self.v - (self.v - self.v_rest) / tau + x
        self.v = v.detach()
        return v

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.soma_dim == 3:
            T, B, C_sub, H, W, N = x_seq.shape
        else:
            T, B, C_sub, H, N = x_seq.shape

        device, dtype = x_seq.device, x_seq.dtype
        dend_input = x_seq

        if not self.no_filter:
            tau_matrix, tau_vec_init, tau_vec_rest, tau = self._build_tau_terms(T, device, dtype)
            if self.decay_input:
                div_tau = tau.view(1, 1, 1, 1, 1, N) if self.soma_dim == 3 else tau.view(1, 1, 1, 1, N)
                dend_input = dend_input / div_tau

            y_branches = []
            for n in range(self.num_branches):
                x_n = dend_input[..., n].reshape(T, -1)
                y_n_flat = torch.matmul(tau_matrix[n], x_n)
                y_n = y_n_flat.view(T, B, C_sub, H, W) if self.soma_dim == 3 else y_n_flat.view(T, B, C_sub, H)

                t_init = tau_vec_init[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_init[n].view(T, 1, 1, 1)
                t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1) if self.soma_dim == 3 else tau_vec_rest[n].view(T, 1, 1, 1)
                v_in = self.v[..., n].detach().unsqueeze(0) if isinstance(self.v, torch.Tensor) else self.v
                y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
                y_branches.append(y_n)

            y = torch.stack(y_branches, dim=-1)
        else:
            y = dend_input

        if self.store_v_seq:
            self.v_seq = y
        self.v = y[-1].detach()

        if self.res:
            y = y + self.skip_weight * x_seq

        if self.bn:
            if self.soma_dim == 3:
                y_permuted = y.permute(0, 1, 5, 2, 3, 4).contiguous()
                y_reshaped = y_permuted.view(T * B, N * C_sub, H, W)
                y_normed = self.branch_bn(y_reshaped).view(T, B, N, C_sub, H, W)
                y_out = y_normed.permute(0, 1, 3, 4, 5, 2).contiguous()
            else:
                y_permuted = y.permute(0, 1, 4, 2, 3).contiguous()
                y_reshaped = y_permuted.view(T * B, N * C_sub, H)
                y_normed = self.branch_bn(y_reshaped).view(T, B, N, C_sub, H)
                y_out = y_normed.permute(0, 1, 3, 4, 2).contiguous()

            if self.last_sigmoid:
                gate = torch.sigmoid(self.gate_alpha * y_out + self.gate_beta)
                y_out = y_out * gate
            if self.last_tanh:
                gate = torch.tanh(self.gate_alpha * y_out + self.gate_beta)
                y_out = y_out * gate
        else:
            y_out = y
            if self.last_sigmoid:
                gate = torch.sigmoid(self.gate_scale * y_out + self.gate_beta2)
                y_out = y_out * gate
            if self.last_tanh:
                gate = torch.tanh(self.gate_scale * y_out + self.gate_beta2)
                y_out = y_out * gate

        #if self.res:
        #    y_out = y_out + self.skip_weight * x_seq

        return y_out

class SoftmaxMixedPureMultiScaleDendCompartment(PureMultiScaleDendCompartment):
    """Pure multi-scale dendrite with normalized learnable branch mixing.

    This class keeps the same branch-wise output shape as
    ``PureMultiScaleDendCompartment``.  It only scales every branch by
    ``softmax(branch_mix_logits)`` before the dendrite-to-soma readout.  When it
    is used with ``VActivationForwardDendNeuron``, the following ``sum(dim=-1)``
    becomes a normalized branch mixture instead of an unnormalized branch sum.
    """

    def __init__(
        self,
        num_branches,
        c_sub=1,
        init_tau=[1.5, 5.0],
        #init_tau=[1.5, 2.0, 4.0, 6.0],
        soma_dim=3,
        decay_input: bool = True,
        bn=True,
        res=False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
        last_sigmoid=False,
        last_tanh=False,
        branch_mix_temperature: float = 1.0,
    ):
        super().__init__(
            num_branches=num_branches,
            c_sub=c_sub,
            init_tau=init_tau,
            soma_dim=soma_dim,
            decay_input=decay_input,
            bn=bn,
            res=res,
            v_rest=v_rest,
            step_mode=step_mode,
            store_v_seq=store_v_seq,
            no_filter=no_filter,
            last_sigmoid=last_sigmoid,
            last_tanh=last_tanh,
        )
        if branch_mix_temperature <= 0:
            raise ValueError("branch_mix_temperature must be positive")
        self.branch_mix_logits = nn.Parameter(torch.zeros(num_branches))
        self.branch_mix_temperature = float(branch_mix_temperature)

    def branch_mix_weights(self, dtype=None, device=None):
        logits = self.branch_mix_logits
        if dtype is not None or device is not None:
            logits = logits.to(dtype=dtype or logits.dtype, device=device or logits.device)
        return torch.softmax(logits / self.branch_mix_temperature, dim=0)

    def _apply_branch_mix(self, y: torch.Tensor):
        weights = self.branch_mix_weights(dtype=y.dtype, device=y.device)
        weights = weights.view(*([1] * (y.dim() - 1)), self.num_branches)
        return y * weights

    def single_step_forward(self, x: torch.Tensor):
        y = super().single_step_forward(x)
        return self._apply_branch_mix(y)

    def multi_step_forward(self, x_seq: torch.Tensor):
        y = super().multi_step_forward(x_seq)
        return self._apply_branch_mix(y)

class SparseRoutedTrunkDistalDendCompartment(BaseDendCompartment):
    """Sparse-routed trunk-distal dendrite for branch scale-up.

    The input compartment dimension is branch-major:
    ``[b0_k0, b0_k1, ..., b1_k0, b1_k1, ...]``.  In every branch, compartment
    ``k0`` is the trunk that carries the signed main signal.  Distal
    compartments form a bounded nonlinear context that modulates the trunk.

    This class deliberately avoids the old BN + ``x * sigmoid(x)`` path.  It is
    designed to make increasing ``num_branches`` add candidate dendritic routes
    rather than unconditionally adding more branch outputs to the soma.
    """

    def __init__(
        self,
        num_branches,
        compartments_per_branch=2,
        c_sub=1,
        init_tau=None,
        tau_min: float = 1.25,
        tau_max: float = 8.0,
        compartment_tau_scale=(1.0, 1.5),
        soma_dim=3,
        decay_input: bool = True,
        selective_update: bool = True,
        active_branches: int = 2,
        route_temperature: float = 1.0,
        route_with_update_gate: bool = True,
        residual_scale: float = 0.1,
        distal_gain_init: float = 0.1,
        update_threshold: float = 0.5,
        update_slope: float = 8.0,
        update_delta_weight: float = 0.5,
        update_activity_weight: float = 0.5,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter=False,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if active_branches <= 0:
            raise ValueError("active_branches must be positive")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        if tau_min <= 1.0 or tau_max <= 1.0:
            raise ValueError("tau_min and tau_max should be larger than 1")

        self.soma_dim = soma_dim
        self.c_sub = c_sub
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.num_compartments = self.num_branches * self.compartments_per_branch
        self.decay_input = decay_input
        self.selective_update = bool(selective_update)
        self.active_branches = min(int(active_branches), self.num_branches)
        self.route_temperature = float(route_temperature)
        self.route_with_update_gate = bool(route_with_update_gate)
        self.no_filter = no_filter
        self.v_rest = v_rest

        tau_data = self._make_tau_data(
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            compartment_tau_scale=compartment_tau_scale,
        )
        self.tau_compartments = nn.Parameter(tau_data.clone().detach().float())

        inv_softplus_one = torch.log(torch.expm1(torch.tensor(1.0)))
        self.distal_shift = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_log_scale = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(inv_softplus_one)))
        self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

        self.route_bias = nn.Parameter(torch.zeros(self.num_branches))
        self.route_score_scale = nn.Parameter(torch.tensor(1.0))
        self.route_update_gain = nn.Parameter(torch.tensor(0.5))

        self.update_threshold = nn.Parameter(torch.full((self.num_branches,), float(update_threshold)))
        self.update_delta_weight = nn.Parameter(torch.tensor(float(update_delta_weight)))
        self.update_activity_weight = nn.Parameter(torch.tensor(float(update_activity_weight)))
        self.update_slope = float(update_slope)

        self.update_gate_seq = None
        self.route_seq = None
        self.distal_energy_seq = None
        self.branch_output_seq = None

    def _make_tau_data(self, init_tau, tau_min, tau_max, compartment_tau_scale):
        if init_tau is not None:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            if tau_data.numel() == 1:
                return torch.full((self.num_compartments,), float(tau_data.flatten()[0]))
            if tau_data.numel() == 2:
                tau_min_v = float(tau_data.flatten()[0])
                tau_max_v = float(tau_data.flatten()[-1])
                branch_tau = torch.exp(torch.linspace(
                    torch.log(torch.tensor(tau_min_v)),
                    torch.log(torch.tensor(tau_max_v)),
                    steps=self.num_branches,
                ))
                comp_scale = torch.linspace(1.0, 1.5, steps=self.compartments_per_branch)
                return (branch_tau[:, None] * comp_scale[None, :]).reshape(-1)
            if tau_data.numel() == self.num_branches:
                return tau_data.flatten().repeat_interleave(self.compartments_per_branch)
            if tau_data.numel() == self.compartments_per_branch:
                return tau_data.flatten().repeat(self.num_branches)
            if tau_data.numel() == self.num_compartments:
                return tau_data.reshape(-1)
            raise ValueError(
                "init_tau must be scalar, length 2, length num_branches, "
                "length compartments_per_branch, or length num_branches * compartments_per_branch"
            )

        branch_tau = torch.exp(torch.linspace(
            torch.log(torch.tensor(float(tau_min))),
            torch.log(torch.tensor(float(tau_max))),
            steps=self.num_branches,
        ))
        if isinstance(compartment_tau_scale, (float, int)):
            comp_scale = torch.full((self.compartments_per_branch,), float(compartment_tau_scale))
        else:
            comp_scale_data = torch.as_tensor(compartment_tau_scale, dtype=torch.float32)
            if comp_scale_data.numel() == 2:
                comp_scale = torch.linspace(
                    float(comp_scale_data.flatten()[0]),
                    float(comp_scale_data.flatten()[-1]),
                    steps=self.compartments_per_branch,
                )
            elif comp_scale_data.numel() == self.compartments_per_branch:
                comp_scale = comp_scale_data.flatten()
            else:
                raise ValueError(
                    "compartment_tau_scale must be scalar, length 2, "
                    "or length compartments_per_branch"
                )
        return (branch_tau[:, None] * comp_scale[None, :]).reshape(-1)

    def _assert_input_shape(self, x: torch.Tensor):
        if x.shape[-1] != self.num_compartments:
            raise ValueError(
                f"Expected last input dimension to be {self.num_compartments} "
                f"(num_branches * compartments_per_branch), but got {x.shape[-1]}"
            )

    def _reshape_to_branches(self, x: torch.Tensor):
        self._assert_input_shape(x)
        return x.reshape(*x.shape[:-1], self.num_branches, self.compartments_per_branch)

    def _view_branch_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 1)), self.num_branches)

    def _view_distal_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(*([1] * (ref.dim() - 2)), self.num_branches, self.distal_count)

    def _compute_update_gate(self, x: torch.Tensor, v_prev: torch.Tensor):
        branch_input = self._reshape_to_branches(x)
        branch_state = self._reshape_to_branches(v_prev)
        if not self.selective_update:
            return torch.ones_like(branch_input[..., 0])

        activity = branch_input.abs().mean(dim=-1)
        delta = (branch_input - branch_state).abs().mean(dim=-1)
        score = self.update_activity_weight.abs() * activity + self.update_delta_weight.abs() * delta
        threshold = self._view_branch_param(self.update_threshold.abs(), score)
        return torch.sigmoid(self.update_slope * (score - threshold))

    def _update_compartment_state(self, x: torch.Tensor):
        self._assert_input_shape(x)
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v)

        v_prev = self.v
        update_gate = self._compute_update_gate(x, v_prev)
        gate = update_gate.unsqueeze(-1).expand(*update_gate.shape, self.compartments_per_branch)
        gate = gate.reshape_as(x)

        if self.no_filter:
            candidate = x
        else:
            tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
            tau = tau.view(*([1] * (x.dim() - 1)), self.num_compartments)
            if self.decay_input:
                candidate = v_prev + (x - (v_prev - self.v_rest)) / tau
            else:
                candidate = v_prev - (v_prev - self.v_rest) / tau + x

        v = (1.0 - gate) * v_prev + gate * candidate
        self.v = v.detach()
        self.update_gate_seq = update_gate
        return v, update_gate

    def _mexican_hat(self, energy: torch.Tensor):
        energy_sq = energy.square()
        return (1.0 - energy_sq) * torch.exp(-0.5 * energy_sq)

    def _compute_branch_output(self, state: torch.Tensor, raw_input: torch.Tensor):
        branch_state = self._reshape_to_branches(state)
        branch_input = self._reshape_to_branches(raw_input)
        trunk = branch_state[..., 0]
        distal = branch_state[..., 1:]

        scale = F.softplus(self._view_distal_param(self.distal_log_scale, distal)) + 1e-4
        shift = self._view_distal_param(self.distal_shift, distal)
        distal_norm = (distal - shift) / scale
        distal_energy = torch.sqrt(distal_norm.square().mean(dim=-1) + 1e-6)
        distal_wave = self._mexican_hat(distal_energy)

        modulation = 1.0 + self._view_branch_param(self.distal_gain, trunk) * distal_wave
        residual = self.residual_scale * branch_input.mean(dim=-1)
        y = self._view_branch_param(self.branch_strength, trunk) * trunk * modulation + residual

        self.distal_energy_seq = distal_energy
        self.branch_output_seq = y
        return y

    def _compute_route(self, branch_output: torch.Tensor, update_gate: torch.Tensor):
        logits = self.route_score_scale.abs() * branch_output.abs()
        logits = logits + self._view_branch_param(self.route_bias, branch_output)
        if self.route_with_update_gate:
            logits = logits + self.route_update_gain * update_gate

        probs = torch.softmax(logits / self.route_temperature, dim=-1)
        if self.active_branches >= self.num_branches:
            route = probs
        else:
            topk_idx = torch.topk(logits, k=self.active_branches, dim=-1).indices
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            route = probs * mask
            route = route / route.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.route_seq = route.detach()
        return route

    def single_step_forward(self, x: torch.Tensor):
        state, update_gate = self._update_compartment_state(x)
        y = self._compute_branch_output(state, x)
        route = self._compute_route(y, update_gate)
        return y * route

    def multi_step_forward(self, x_seq: torch.Tensor):
        self._assert_input_shape(x_seq)
        y_seq = []
        update_gate_seq = []
        route_seq = []
        distal_energy_seq = []
        branch_output_seq = []
        v_seq = [] if self.store_v_seq else None

        for t in range(x_seq.shape[0]):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            update_gate_seq.append(self.update_gate_seq)
            route_seq.append(self.route_seq)
            distal_energy_seq.append(self.distal_energy_seq)
            branch_output_seq.append(self.branch_output_seq)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        self.update_gate_seq = torch.stack(update_gate_seq)
        self.route_seq = torch.stack(route_seq)
        self.distal_energy_seq = torch.stack(distal_energy_seq)
        self.branch_output_seq = torch.stack(branch_output_seq)
        return torch.stack(y_seq)

class ChannelPreservingTrunkDistalDendCompartment(BaseDendCompartment):
    """Channel-preserving trunk-distal dendrite.

    This module is intended for the ``C_out -> C_out`` dendrite design.  Unlike
    ``SegregatedDend``-style compartments, it does not require the preceding
    Conv/Linear layer to emit ``C * num_branches * compartments_per_branch``
    channels.  The public input and output shapes are the same:

    - single-step 1D: ``[N, C, L] -> [N, C, L]``
    - single-step 2D: ``[N, C, H, W] -> [N, C, H, W]``
    - multi-step 1D: ``[T, N, C, L] -> [T, N, C, L]``
    - multi-step 2D: ``[T, N, C, H, W] -> [T, N, C, H, W]``

    Branches and compartments are internal dimensions.  Each channel is assigned
    to a small number of branches with an overlapping fixed mask, then each
    branch keeps ``K`` compartment states.  The branch readout uses a trunk signal
    modulated by distal compartment differences, rather than a linear sum over
    compartments.
    """

    def __init__(
        self,
        channels=None,
        num_branches=4,
        compartments_per_branch=2,
        c_sub=None,
        branch_degree=2,
        init_tau=None,
        tau_min: float = 1.25,
        tau_max: float = 8.0,
        compartment_tau_scale=(1.0, 1.5),
        decay_input: bool = True,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter: bool = False,
        merge_norm: str = "sqrt",
        learn_channel_gate: bool = True,
        channel_gate_scale: float = 0.5,
        distal_gain_init: float = 0.1,
        distal_residual_init: float = 0.0,
        distal_gate_alpha_init: float = 6.0,
        distal_gate_threshold_init: float = 0.5,
        input_residual_init: float = 0.0,
        detach_state_during_forward: bool = False,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if channels is None:
            channels = c_sub
        if channels is None:
            raise ValueError("channels or c_sub must be provided")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if branch_degree <= 0:
            raise ValueError("branch_degree must be positive")
        if merge_norm not in {"sqrt", "mean", "sum"}:
            raise ValueError("merge_norm must be one of: 'sqrt', 'mean', 'sum'")
        if tau_min <= 1.0 or tau_max <= 1.0:
            raise ValueError("tau_min and tau_max should be larger than 1")

        self.channels = int(channels)
        self.c_sub = self.channels
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.branch_degree = min(int(branch_degree), self.num_branches)
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.no_filter = no_filter
        self.merge_norm = merge_norm
        self.learn_channel_gate = bool(learn_channel_gate)
        self.channel_gate_scale = float(channel_gate_scale)
        self.detach_state_during_forward = bool(detach_state_during_forward)

        branch_mask = self._make_branch_mask(self.channels, self.num_branches, self.branch_degree)
        channel_degree = branch_mask.sum(dim=0).clamp_min(1.0)
        self.register_buffer("branch_mask", branch_mask)
        self.register_buffer("channel_degree", channel_degree)

        tau_data = self._make_tau_data(
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            compartment_tau_scale=compartment_tau_scale,
        )
        self.tau_compartments = nn.Parameter(tau_data.clone().detach().float())

        self.branch_channel_gain = nn.Parameter(torch.zeros(self.num_branches, self.channels))
        self.compartment_input_gain = nn.Parameter(torch.zeros(self.num_branches, self.compartments_per_branch))
        self.distal_mix_logits = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_gate_alpha = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_alpha_init)))
        self.distal_gate_threshold = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_threshold_init)))
        self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
        self.distal_residual_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_residual_init)))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        self.input_residual_scale = nn.Parameter(torch.tensor(float(input_residual_init)))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.branch_mod_seq = None
        self.distal_gate_seq = None
        self.branch_output_seq = None
        self.branch_input_seq = None

    @staticmethod
    def _make_branch_mask(channels: int, num_branches: int, branch_degree: int):
        mask = torch.zeros(num_branches, channels, dtype=torch.float32)
        channel_index = torch.arange(channels)
        for offset in range(branch_degree):
            branch_index = (channel_index + offset) % num_branches
            mask[branch_index, channel_index] = 1.0
        return mask

    def _make_tau_data(self, init_tau, tau_min, tau_max, compartment_tau_scale):
        if init_tau is not None:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            if tau_data.numel() == 1:
                return torch.full((self.num_branches, self.compartments_per_branch), float(tau_data.flatten()[0]))
            if tau_data.numel() == 2:
                tau_min_v = float(tau_data.flatten()[0])
                tau_max_v = float(tau_data.flatten()[-1])
                branch_tau = torch.exp(torch.linspace(
                    torch.log(torch.tensor(tau_min_v)),
                    torch.log(torch.tensor(tau_max_v)),
                    steps=self.num_branches,
                ))
                comp_scale = torch.linspace(1.0, 1.5, steps=self.compartments_per_branch)
                return branch_tau[:, None] * comp_scale[None, :]
            if tau_data.numel() == self.num_branches:
                return tau_data.flatten()[:, None].repeat(1, self.compartments_per_branch)
            if tau_data.numel() == self.compartments_per_branch:
                return tau_data.flatten()[None, :].repeat(self.num_branches, 1)
            if tau_data.numel() == self.num_branches * self.compartments_per_branch:
                return tau_data.reshape(self.num_branches, self.compartments_per_branch)
            raise ValueError(
                "init_tau must be scalar, length 2, length num_branches, "
                "length compartments_per_branch, or length num_branches * compartments_per_branch"
            )

        branch_tau = torch.exp(torch.linspace(
            torch.log(torch.tensor(float(tau_min))),
            torch.log(torch.tensor(float(tau_max))),
            steps=self.num_branches,
        ))
        if isinstance(compartment_tau_scale, (float, int)):
            comp_scale = torch.full((self.compartments_per_branch,), float(compartment_tau_scale))
        else:
            comp_scale_data = torch.as_tensor(compartment_tau_scale, dtype=torch.float32)
            if comp_scale_data.numel() == 2:
                comp_scale = torch.linspace(
                    float(comp_scale_data.flatten()[0]),
                    float(comp_scale_data.flatten()[-1]),
                    steps=self.compartments_per_branch,
                )
            elif comp_scale_data.numel() == self.compartments_per_branch:
                comp_scale = comp_scale_data.flatten()
            else:
                raise ValueError(
                    "compartment_tau_scale must be scalar, length 2, "
                    "or length compartments_per_branch"
                )
        return branch_tau[:, None] * comp_scale[None, :]

    def _assert_input_shape(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError("Expected input shape [N, C, ...]")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x.shape[1]}")

    def _view_branch_channel(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, self.channels, 1, *([1] * (ref.dim() - 4)))

    def _view_branch_compartment(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, 1, self.compartments_per_branch, *([1] * (ref.dim() - 4)))

    def _view_distal_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, 1, self.distal_count, *([1] * (ref.dim() - 4)))

    def _view_branch_param(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.num_branches, 1, *([1] * (ref.dim() - 3)))

    def _view_channel_degree(self, ref: torch.Tensor):
        return self.channel_degree.to(dtype=ref.dtype, device=ref.device).view(
            1, self.channels, *([1] * (ref.dim() - 2))
        )

    def _branch_channel_weight(self, dtype, device):
        weight = self.branch_mask.to(dtype=dtype, device=device)
        if self.learn_channel_gate:
            gate = 1.0 + self.channel_gate_scale * torch.tanh(self.branch_channel_gain.to(dtype=dtype, device=device))
            weight = weight * gate
        return weight

    def _build_branch_input(self, x: torch.Tensor):
        self._assert_input_shape(x)
        x_branch = x.unsqueeze(1).unsqueeze(3)
        branch_weight = self._branch_channel_weight(x.dtype, x.device)
        branch_weight = self._view_branch_channel(branch_weight, x_branch)
        comp_gain = 1.0 + 0.5 * torch.tanh(self.compartment_input_gain.to(dtype=x.dtype, device=x.device))
        comp_gain = self._view_branch_compartment(comp_gain, x_branch)
        branch_input = x_branch * branch_weight * comp_gain
        self.branch_input_seq = branch_input.detach()
        return branch_input

    def _init_state(self, branch_input: torch.Tensor):
        if isinstance(self.v, float):
            return torch.full_like(branch_input, self.v)
        if self.v.shape != branch_input.shape:
            return torch.full_like(branch_input, self.v_rest)
        return self.v

    def _integrate(self, branch_input: torch.Tensor, v_prev: torch.Tensor):
        if self.no_filter:
            return branch_input

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._view_branch_compartment(tau.to(dtype=branch_input.dtype, device=branch_input.device), branch_input)
        if self.decay_input:
            return v_prev + (branch_input - (v_prev - self.v_rest)) / tau
        return v_prev - (v_prev - self.v_rest) / tau + branch_input

    def _normalize_distal_delta(self, delta: torch.Tensor):
        reduce_dims = [2] + list(range(4, delta.dim()))
        rms = torch.sqrt(delta.square().mean(dim=reduce_dims, keepdim=True) + 1e-6)
        return delta / rms

    def _compute_branch_output(self, state: torch.Tensor):
        trunk = state[:, :, :, 0, ...]
        distal = state[:, :, :, 1:, ...]
        delta = distal - trunk.unsqueeze(3)
        delta_norm = self._normalize_distal_delta(delta)

        alpha = F.softplus(self._view_distal_param(self.distal_gate_alpha, delta_norm))
        threshold = self._view_distal_param(self.distal_gate_threshold.abs(), delta_norm)
        distal_gate = torch.sigmoid(alpha * delta_norm.abs() - threshold)

        mix = torch.softmax(self.distal_mix_logits, dim=-1)
        mix = self._view_distal_param(mix.to(dtype=state.dtype, device=state.device), delta_norm)
        branch_mod = (mix * distal_gate * torch.tanh(delta_norm)).sum(dim=3)
        branch_mod = torch.tanh(branch_mod)

        distal_gain = self._view_branch_param(self.distal_gain, trunk)
        residual_gain = self._view_branch_param(self.distal_residual_gain, trunk)
        branch_strength = self._view_branch_param(self.branch_strength, trunk)
        y = branch_strength * (trunk + distal_gain * trunk * branch_mod + residual_gain * branch_mod)

        self.branch_mod_seq = branch_mod.detach()
        self.distal_gate_seq = distal_gate.detach()
        self.branch_output_seq = y.detach()
        return y

    def _merge_branches(self, branch_output: torch.Tensor, raw_input: torch.Tensor):
        y = branch_output.sum(dim=1)
        degree = self._view_channel_degree(y)
        if self.merge_norm == "sqrt":
            y = y / torch.sqrt(degree)
        elif self.merge_norm == "mean":
            y = y / degree
        y = self.output_scale * y
        return y + self.input_residual_scale * raw_input

    def _step(self, x: torch.Tensor, v_prev: torch.Tensor):
        branch_input = self._build_branch_input(x)
        state = self._integrate(branch_input, v_prev)
        branch_output = self._compute_branch_output(state)
        y = self._merge_branches(branch_output, x)
        return y, state

    def single_step_forward(self, x: torch.Tensor):
        branch_input = self._build_branch_input(x)
        v_prev = self._init_state(branch_input)
        state = self._integrate(branch_input, v_prev)
        branch_output = self._compute_branch_output(state)
        y = self._merge_branches(branch_output, x)
        self.v = state.detach()
        if self.store_v_seq:
            self.v_seq = state.detach().unsqueeze(0)
        return y

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() < 3:
            raise ValueError("Expected multi-step input shape [T, N, C, ...]")
        if x_seq.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x_seq.shape[2]}")

        y_seq = []
        v_seq = [] if self.store_v_seq else None
        branch_mod_seq = []
        distal_gate_seq = []
        branch_output_seq = []

        first_branch_input = self._build_branch_input(x_seq[0])
        v = self._init_state(first_branch_input)
        for t in range(x_seq.shape[0]):
            if t == 0:
                branch_input = first_branch_input
                state = self._integrate(branch_input, v)
                branch_output = self._compute_branch_output(state)
                y = self._merge_branches(branch_output, x_seq[t])
            else:
                y, state = self._step(x_seq[t], v)
            y_seq.append(y)
            branch_mod_seq.append(self.branch_mod_seq)
            distal_gate_seq.append(self.distal_gate_seq)
            branch_output_seq.append(self.branch_output_seq)
            if self.store_v_seq:
                v_seq.append(state.detach())
            v = state.detach() if self.detach_state_during_forward else state

        self.v = v.detach()
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        self.branch_mod_seq = torch.stack(branch_mod_seq)
        self.distal_gate_seq = torch.stack(distal_gate_seq)
        self.branch_output_seq = torch.stack(branch_output_seq)
        return torch.stack(y_seq)

class SparseChannelPreservingTrunkDistalDendCompartment(BaseDendCompartment):
    """Active-edge channel-preserving trunk-distal dendrite.

    This is the sparse version of ``ChannelPreservingTrunkDistalDendCompartment``.
    The external mapping is still ``C -> C``, but the internal state is
    ``[N, d, C, K, ...]`` instead of ``[N, B, C, K, ...]``.  Here ``B`` is the
    size of the branch parameter library and ``d`` is ``branch_degree``.  Scaling
    ``num_branches`` therefore adds candidate branch dynamics without forcing
    every channel to materialize every branch.
    """

    def __init__(
        self,
        channels=None,
        num_branches=4,
        compartments_per_branch=2,
        c_sub=None,
        branch_degree=2,
        branch_assignment: str = "cyclic",
        init_tau=None,
        tau_min: float = 1.25,
        tau_max: float = 5.0,
        compartment_tau_scale=(1.0, 2.5),
        decay_input: bool = True,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        store_branch_monitor: bool = False,
        no_filter: bool = False,
        merge_norm: str = "sqrt",
        learn_edge_gain: bool = True,
        edge_gain_scale: float = 0.5,
        distal_gain_init: float = 0.1,
        distal_residual_init: float = 0.0,
        distal_gate_alpha_init: float = 6.0,
        distal_gate_threshold_init: float = 0.5,
        input_residual_init: float = 0.0,
        detach_state_during_forward: bool = False,
        parallel_forward: bool = True,
    ):
        super().__init__(v_rest, step_mode, store_v_seq)
        if channels is None:
            channels = c_sub
        if channels is None:
            raise ValueError("channels or c_sub must be provided")
        if int(channels) <= 0:
            raise ValueError("channels must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if compartments_per_branch < 2:
            raise ValueError("compartments_per_branch must be at least 2")
        if branch_degree <= 0:
            raise ValueError("branch_degree must be positive")
        if branch_assignment not in {"window", "cyclic"}:
            raise ValueError("branch_assignment must be 'window' or 'cyclic'")
        if merge_norm not in {"sqrt", "mean", "sum"}:
            raise ValueError("merge_norm must be one of: 'sqrt', 'mean', 'sum'")
        if tau_min <= 1.0 or tau_max <= 1.0:
            raise ValueError("tau_min and tau_max should be larger than 1")

        self.channels = int(channels)
        self.c_sub = self.channels
        self.num_branches = int(num_branches)
        self.compartments_per_branch = int(compartments_per_branch)
        self.distal_count = self.compartments_per_branch - 1
        self.branch_degree = min(int(branch_degree), self.num_branches)
        self.branch_assignment = branch_assignment
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.no_filter = no_filter
        self.merge_norm = merge_norm
        self.learn_edge_gain = bool(learn_edge_gain)
        self.edge_gain_scale = float(edge_gain_scale)
        self.detach_state_during_forward = bool(detach_state_during_forward)
        self.parallel_forward = bool(parallel_forward)
        self.store_branch_monitor = bool(store_branch_monitor)

        active_branch_index = self._make_active_branch_index(
            self.channels,
            self.num_branches,
            self.branch_degree,
            self.branch_assignment,
        )
        channel_degree = torch.full((self.channels,), float(self.branch_degree))
        branch_usage = torch.bincount(
            active_branch_index.reshape(-1),
            minlength=self.num_branches,
        ).float()
        self.register_buffer("active_branch_index", active_branch_index)
        self.register_buffer("channel_degree", channel_degree)
        self.register_buffer("branch_usage", branch_usage)

        tau_data = self._make_tau_data(
            init_tau=init_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            compartment_tau_scale=compartment_tau_scale,
        )
        self.tau_compartments = nn.Parameter(tau_data.clone().detach().float())

        if self.learn_edge_gain:
            self.edge_channel_gain = nn.Parameter(torch.zeros(self.branch_degree, self.channels))
        else:
            self.register_buffer("edge_channel_gain", torch.zeros(self.branch_degree, self.channels))

        self.compartment_input_logits = nn.Parameter(torch.zeros(self.num_branches, self.compartments_per_branch))
        self.distal_mix_logits = nn.Parameter(torch.zeros(self.num_branches, self.distal_count))
        self.distal_gate_alpha = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_alpha_init)))
        self.distal_gate_threshold = nn.Parameter(torch.full((self.num_branches, self.distal_count), float(distal_gate_threshold_init)))
        self.distal_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_gain_init)))
        self.distal_residual_gain = nn.Parameter(torch.full((self.num_branches,), float(distal_residual_init)))
        self.branch_strength = nn.Parameter(torch.ones(self.num_branches))
        self.input_residual_scale = nn.Parameter(torch.tensor(float(input_residual_init)))
        self.output_scale = nn.Parameter(torch.tensor(1.0))

        self.branch_input_seq = None
        self.branch_mod_seq = None
        self.distal_gate_seq = None
        self.branch_output_seq = None

    @staticmethod
    def _make_active_branch_index(
        channels: int,
        num_branches: int,
        branch_degree: int,
        branch_assignment: str,
    ):
        channel_index = torch.arange(channels, dtype=torch.long)
        if branch_assignment == "window":
            base = torch.div(channel_index * num_branches, channels, rounding_mode="floor")
        else:
            base = channel_index % num_branches

        edges = []
        for offset in range(branch_degree):
            edges.append((base + offset) % num_branches)
        return torch.stack(edges, dim=0)

    def _make_tau_data(self, init_tau, tau_min, tau_max, compartment_tau_scale):
        if init_tau is not None:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            if tau_data.numel() == 1:
                return torch.full((self.num_branches, self.compartments_per_branch), float(tau_data.flatten()[0]))
            if tau_data.numel() == 2:
                tau_min_v = float(tau_data.flatten()[0])
                tau_max_v = float(tau_data.flatten()[-1])
                branch_tau = torch.exp(torch.linspace(
                    torch.log(torch.tensor(tau_min_v)),
                    torch.log(torch.tensor(tau_max_v)),
                    steps=self.num_branches,
                ))
                comp_scale = torch.linspace(1.0, 1.5, steps=self.compartments_per_branch)
                return branch_tau[:, None] * comp_scale[None, :]
            if tau_data.numel() == self.num_branches:
                return tau_data.flatten()[:, None].repeat(1, self.compartments_per_branch)
            if tau_data.numel() == self.compartments_per_branch:
                return tau_data.flatten()[None, :].repeat(self.num_branches, 1)
            if tau_data.numel() == self.num_branches * self.compartments_per_branch:
                return tau_data.reshape(self.num_branches, self.compartments_per_branch)
            raise ValueError(
                "init_tau must be scalar, length 2, length num_branches, "
                "length compartments_per_branch, or length num_branches * compartments_per_branch"
            )

        branch_tau = torch.exp(torch.linspace(
            torch.log(torch.tensor(float(tau_min))),
            torch.log(torch.tensor(float(tau_max))),
            steps=self.num_branches,
        ))
        if isinstance(compartment_tau_scale, (float, int)):
            comp_scale = torch.full((self.compartments_per_branch,), float(compartment_tau_scale))
        else:
            comp_scale_data = torch.as_tensor(compartment_tau_scale, dtype=torch.float32)
            if comp_scale_data.numel() == 2:
                comp_scale = torch.linspace(
                    float(comp_scale_data.flatten()[0]),
                    float(comp_scale_data.flatten()[-1]),
                    steps=self.compartments_per_branch,
                )
            elif comp_scale_data.numel() == self.compartments_per_branch:
                comp_scale = comp_scale_data.flatten()
            else:
                raise ValueError(
                    "compartment_tau_scale must be scalar, length 2, "
                    "or length compartments_per_branch"
                )
        return branch_tau[:, None] * comp_scale[None, :]

    def _assert_input_shape(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError("Expected input shape [N, C, ...]")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x.shape[1]}")

    def _edge_index(self, device=None):
        if device is None:
            return self.active_branch_index
        return self.active_branch_index.to(device=device)

    def _edge_branch_value(self, value: torch.Tensor):
        return value[self._edge_index(value.device)]

    def _view_edge_channel(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.branch_degree, self.channels, *([1] * (ref.dim() - 3)))

    def _view_edge_compartment(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.branch_degree, self.channels, self.compartments_per_branch, *([1] * (ref.dim() - 4)))

    def _view_edge_distal(self, value: torch.Tensor, ref: torch.Tensor):
        return value.view(1, self.branch_degree, self.channels, self.distal_count, *([1] * (ref.dim() - 4)))

    def _view_channel_degree(self, ref: torch.Tensor):
        return self.channel_degree.to(dtype=ref.dtype, device=ref.device).view(
            1, self.channels, *([1] * (ref.dim() - 2))
        )

    def _edge_channel_weight(self, dtype, device):
        gain = self.edge_channel_gain.to(dtype=dtype, device=device)
        if self.learn_edge_gain:
            return 1.0 + self.edge_gain_scale * torch.tanh(gain)
        return torch.ones_like(gain)

    def _edge_compartment_input_gain(self, dtype, device):
        raw = F.softplus(self.compartment_input_logits.to(dtype=dtype, device=device)) + 1e-4
        normalized = self.compartments_per_branch * raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self._edge_branch_value(normalized)

    def _build_branch_input(self, x: torch.Tensor):
        self._assert_input_shape(x)
        x_edge = x.unsqueeze(1).unsqueeze(3)
        edge_gain = self._view_edge_channel(self._edge_channel_weight(x.dtype, x.device), x.unsqueeze(1))
        comp_gain = self._view_edge_compartment(self._edge_compartment_input_gain(x.dtype, x.device), x_edge)
        branch_input = x_edge * edge_gain.unsqueeze(3) * comp_gain
        if self.store_branch_monitor:
            self._branch_input_step = branch_input.detach()
        else:
            self._branch_input_step = None
        return branch_input

    def _init_state(self, branch_input: torch.Tensor):
        if isinstance(self.v, float):
            return torch.full_like(branch_input, self.v)
        if self.v.shape != branch_input.shape:
            return torch.full_like(branch_input, self.v_rest)
        return self.v

    def _integrate(self, branch_input: torch.Tensor, v_prev: torch.Tensor):
        if self.no_filter:
            return branch_input

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._edge_branch_value(tau.to(dtype=branch_input.dtype, device=branch_input.device))
        tau = self._view_edge_compartment(tau, branch_input)
        if self.decay_input:
            return v_prev + (branch_input - (v_prev - self.v_rest)) / tau
        return v_prev - (v_prev - self.v_rest) / tau + branch_input

    def _build_branch_input_sequence(self, x_seq: torch.Tensor):
        x_edge = x_seq.unsqueeze(2).unsqueeze(4)
        spatial_ones = [1] * (x_seq.dim() - 3)
        edge_gain = self._edge_channel_weight(x_seq.dtype, x_seq.device).view(
            1, 1, self.branch_degree, self.channels, 1, *spatial_ones
        )
        comp_gain = self._edge_compartment_input_gain(x_seq.dtype, x_seq.device).view(
            1, 1, self.branch_degree, self.channels, self.compartments_per_branch, *spatial_ones
        )
        return x_edge * edge_gain * comp_gain

    def _build_tau_terms(self, tau: torch.Tensor, T: int, dtype, device):
        tau = torch.clamp(tau.reshape(-1).to(dtype=dtype, device=device), min=1.0 + 1e-3)
        decay = 1.0 - 1.0 / tau
        t = torch.arange(T, dtype=dtype, device=device)
        diff = t[:, None] - t[None, :]
        mask = diff >= 0
        powers = decay[:, None, None] ** diff.clamp_min(0).unsqueeze(0)
        tau_matrix = torch.where(
            mask.unsqueeze(0),
            powers,
            torch.zeros((), dtype=dtype, device=device),
        )
        tau_vec_init = decay[:, None] ** (t + 1.0)
        tau_vec_rest = 1.0 - tau_vec_init
        return tau_matrix, tau_vec_init, tau_vec_rest, tau

    def _parallel_integrate(self, branch_input_seq: torch.Tensor, v_init: torch.Tensor):
        if self.no_filter:
            return branch_input_seq

        T, N = branch_input_seq.shape[:2]
        spatial_shape = branch_input_seq.shape[5:]
        flat_size = self.branch_degree * self.channels * self.compartments_per_branch

        permute_to_edge_first = (2, 3, 4, 0, 1) + tuple(range(5, branch_input_seq.dim()))
        branch_input_flat = branch_input_seq.permute(permute_to_edge_first).reshape(flat_size, T, -1)

        v_permute_to_edge_first = (1, 2, 3, 0) + tuple(range(4, v_init.dim()))
        v_init_flat = v_init.permute(v_permute_to_edge_first).reshape(flat_size, -1)

        tau = torch.clamp(self.tau_compartments, min=1.0 + 1e-3)
        tau = self._edge_branch_value(tau.to(dtype=branch_input_seq.dtype, device=branch_input_seq.device))
        tau_matrix, tau_vec_init, tau_vec_rest, tau_flat = self._build_tau_terms(
            tau,
            T,
            branch_input_seq.dtype,
            branch_input_seq.device,
        )

        if self.decay_input:
            drive = branch_input_flat / tau_flat[:, None, None]
        else:
            drive = branch_input_flat

        state_flat = torch.bmm(tau_matrix, drive)
        state_flat = state_flat + tau_vec_init[:, :, None] * v_init_flat[:, None, :]
        state_flat = state_flat + tau_vec_rest[:, :, None] * self.v_rest

        edge_first_shape = (
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            T,
            N,
            *spatial_shape,
        )
        state_edge_first = state_flat.reshape(edge_first_shape)
        permute_to_time_first = (3, 4, 0, 1, 2) + tuple(range(5, len(edge_first_shape)))
        return state_edge_first.permute(permute_to_time_first).contiguous()

    def _normalize_distal_delta(self, delta: torch.Tensor):
        reduce_dims = [2] + list(range(4, delta.dim()))
        rms = torch.sqrt(delta.square().mean(dim=reduce_dims, keepdim=True) + 1e-6)
        return delta / rms

    def _compute_branch_output(self, state: torch.Tensor):
        trunk = state[:, :, :, 0, ...]
        distal = state[:, :, :, 1:, ...]
        delta = distal - trunk.unsqueeze(3)
        delta_norm = self._normalize_distal_delta(delta)

        alpha = F.softplus(self._edge_branch_value(self.distal_gate_alpha).to(dtype=state.dtype, device=state.device))
        threshold = self._edge_branch_value(self.distal_gate_threshold.abs()).to(dtype=state.dtype, device=state.device)
        alpha = self._view_edge_distal(alpha, delta_norm)
        threshold = self._view_edge_distal(threshold, delta_norm)
        distal_gate = torch.sigmoid(alpha * delta_norm.abs() - threshold)

        mix = torch.softmax(self.distal_mix_logits, dim=-1)
        mix = self._edge_branch_value(mix.to(dtype=state.dtype, device=state.device))
        mix = self._view_edge_distal(mix, delta_norm)
        branch_mod = (mix * distal_gate * torch.tanh(delta_norm)).sum(dim=3)
        branch_mod = torch.tanh(branch_mod)

        distal_gain = self._edge_branch_value(self.distal_gain).to(dtype=state.dtype, device=state.device)
        residual_gain = self._edge_branch_value(self.distal_residual_gain).to(dtype=state.dtype, device=state.device)
        branch_strength = self._edge_branch_value(self.branch_strength).to(dtype=state.dtype, device=state.device)
        distal_gain = self._view_edge_channel(distal_gain, trunk)
        residual_gain = self._view_edge_channel(residual_gain, trunk)
        branch_strength = self._view_edge_channel(branch_strength, trunk)

        y = branch_strength * (trunk + distal_gain * trunk * branch_mod + residual_gain * branch_mod)

        if self.store_branch_monitor:
            self._branch_mod_step = branch_mod.detach()
            self._distal_gate_step = distal_gate.detach()
            self._branch_output_step = y.detach()
        else:
            self._branch_mod_step = None
            self._distal_gate_step = None
            self._branch_output_step = None
        return y

    def _merge_edges(self, edge_output: torch.Tensor, raw_input: torch.Tensor):
        y = edge_output.sum(dim=1)
        degree = self._view_channel_degree(y)
        if self.merge_norm == "sqrt":
            y = y / torch.sqrt(degree)
        elif self.merge_norm == "mean":
            y = y / degree
        return self.output_scale * y + self.input_residual_scale * raw_input

    def _compute_branch_output_sequence(self, state_seq: torch.Tensor):
        T, N = state_seq.shape[:2]
        spatial_shape = state_seq.shape[5:]
        state_flat = state_seq.reshape(
            T * N,
            self.branch_degree,
            self.channels,
            self.compartments_per_branch,
            *spatial_shape,
        )
        edge_output_flat = self._compute_branch_output(state_flat)
        edge_output_seq = edge_output_flat.reshape(
            T,
            N,
            self.branch_degree,
            self.channels,
            *spatial_shape,
        )
        if self.store_branch_monitor:
            self.branch_mod_seq = self._branch_mod_step.reshape(
                T,
                N,
                self.branch_degree,
                self.channels,
                *spatial_shape,
            )
            self.distal_gate_seq = self._distal_gate_step.reshape(
                T,
                N,
                self.branch_degree,
                self.channels,
                self.distal_count,
                *spatial_shape,
            )
            self.branch_output_seq = self._branch_output_step.reshape(
                T,
                N,
                self.branch_degree,
                self.channels,
                *spatial_shape,
            )
        return edge_output_seq

    def _merge_edges_sequence(self, edge_output_seq: torch.Tensor, raw_input_seq: torch.Tensor):
        T, N = raw_input_seq.shape[:2]
        spatial_shape = raw_input_seq.shape[3:]
        edge_output_flat = edge_output_seq.reshape(
            T * N,
            self.branch_degree,
            self.channels,
            *spatial_shape,
        )
        raw_input_flat = raw_input_seq.reshape(T * N, self.channels, *spatial_shape)
        y_flat = self._merge_edges(edge_output_flat, raw_input_flat)
        return y_flat.reshape(T, N, self.channels, *spatial_shape)

    def _step(self, x: torch.Tensor, v_prev: torch.Tensor):
        branch_input = self._build_branch_input(x)
        state = self._integrate(branch_input, v_prev)
        edge_output = self._compute_branch_output(state)
        y = self._merge_edges(edge_output, x)
        return y, state

    def single_step_forward(self, x: torch.Tensor):
        branch_input = self._build_branch_input(x)
        v_prev = self._init_state(branch_input)
        state = self._integrate(branch_input, v_prev)
        edge_output = self._compute_branch_output(state)
        y = self._merge_edges(edge_output, x)
        self.v = state.detach()
        if self.store_v_seq:
            self.v_seq = state.detach().unsqueeze(0)
        if self.store_branch_monitor:
            self.branch_input_seq = self._branch_input_step
            self.branch_mod_seq = self._branch_mod_step
            self.distal_gate_seq = self._distal_gate_step
            self.branch_output_seq = self._branch_output_step
        else:
            self.branch_input_seq = None
            self.branch_mod_seq = None
            self.distal_gate_seq = None
            self.branch_output_seq = None
        return y

    def _sequential_multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() < 3:
            raise ValueError("Expected multi-step input shape [T, N, C, ...]")
        if x_seq.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x_seq.shape[2]}")

        y_seq = []
        v_seq = [] if self.store_v_seq else None
        if self.store_branch_monitor:
            branch_input_seq = []
            branch_mod_seq = []
            distal_gate_seq = []
            branch_output_seq = []

        first_branch_input = self._build_branch_input(x_seq[0])
        v = self._init_state(first_branch_input)
        for t in range(x_seq.shape[0]):
            if t == 0:
                state = self._integrate(first_branch_input, v)
                edge_output = self._compute_branch_output(state)
                y = self._merge_edges(edge_output, x_seq[t])
            else:
                y, state = self._step(x_seq[t], v)
            y_seq.append(y)

            if self.store_branch_monitor:
                branch_input_seq.append(self._branch_input_step)
                branch_mod_seq.append(self._branch_mod_step)
                distal_gate_seq.append(self._distal_gate_step)
                branch_output_seq.append(self._branch_output_step)
            if self.store_v_seq:
                v_seq.append(state.detach())
            v = state.detach() if self.detach_state_during_forward else state

        self.v = v.detach()
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_branch_monitor:
            self.branch_input_seq = torch.stack(branch_input_seq)
            self.branch_mod_seq = torch.stack(branch_mod_seq)
            self.distal_gate_seq = torch.stack(distal_gate_seq)
            self.branch_output_seq = torch.stack(branch_output_seq)
        else:
            self.branch_input_seq = None
            self.branch_mod_seq = None
            self.distal_gate_seq = None
            self.branch_output_seq = None
        return torch.stack(y_seq)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.dim() < 3:
            raise ValueError("Expected multi-step input shape [T, N, C, ...]")
        if x_seq.shape[2] != self.channels:
            raise ValueError(f"Expected {self.channels} input channels, but got {x_seq.shape[2]}")
        if (not self.parallel_forward) or self.detach_state_during_forward:
            return self._sequential_multi_step_forward(x_seq)

        branch_input_seq = self._build_branch_input_sequence(x_seq)
        v_init = self._init_state(branch_input_seq[0])
        state_seq = self._parallel_integrate(branch_input_seq, v_init)
        edge_output_seq = self._compute_branch_output_sequence(state_seq)
        y_seq = self._merge_edges_sequence(edge_output_seq, x_seq)

        self.v = state_seq[-1].detach()
        if self.store_v_seq:
            self.v_seq = state_seq.detach()
        if self.store_branch_monitor:
            self.branch_input_seq = branch_input_seq.detach()
            self._branch_input_step = self.branch_input_seq[-1]
            self._branch_mod_step = self.branch_mod_seq[-1]
            self._distal_gate_step = self.distal_gate_seq[-1]
            self._branch_output_step = self.branch_output_seq[-1]
        else:
            self.branch_input_seq = None
            self.branch_mod_seq = None
            self.distal_gate_seq = None
            self.branch_output_seq = None
            self._branch_input_step = None
            self._branch_mod_step = None
            self._distal_gate_step = None
            self._branch_output_step = None
        return y_seq

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
