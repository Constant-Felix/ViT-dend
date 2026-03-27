"""The voltage dynamics of dendritic compartments.

This package contains a series of classes depicting different types of dendritic
compartments so that we can compute the dendritic voltage dynamics step by step,
given the input to all the compartments. When computing the dendritic voltage 
dynamics for a single time step, the compartments are treated independently. 
The relationship (wiring) among a set of compartments is not considered here.
"""

import abc
from typing import Callable

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
        tau: float = 1.01,
        decay_input: bool = False,  ###
        gating = True,
        res = False,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        no_filter = False
    ):
        super().__init__(v_rest, step_mode, store_v_seq)

        # ----------------------------------------------------------------
        # 保持 Tau 为标量 (1,)，对所有分支共享
        # ----------------------------------------------------------------
        #self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.gating = gating
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.skip_weight = nn.Parameter(torch.tensor(1.0))
        self.res = res
        self.no_filter = no_filter

        # 2. 门控参数 (仅当 gating=True 时创建)
        if self.gating:
            # 这是一个可学习的缩放因子，替代 BN 的 weight
            # 初始化小一点 (0.2)，因为 decay_input=False 时积分值会很大
            # 我们需要把它缩放到 sigmoid 敏感区 (-5 ~ 5)
            self.gate_scale = nn.Parameter(torch.tensor(0.5)) 
            
            # 控制门控的偏置，初始化为 0，不引入背景噪声
            self.gate_beta = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    # 构建因果并行矩阵 (Lower Triangular)
    # ------------------------------------------------------------------
    def _build_tau_terms(self, T: int, device, dtype):
        """
        Builds scalar tau-dependent matrices.
        Returns:
            tau_matrix: (T, T) -> Lower Triangular (Causal)
            tau_vec_init: (T,)
            tau_vec_rest: (T,)
            tau: Scalar
        """
        # scalar clamp
        tau = torch.clamp(self.tau, min=1.0 + 1e-3)
        a = 1.0 - 1.0 / tau

        t = torch.arange(T, device=device, dtype=dtype)

        # ------------------------------------------------------------
        # Matrix Construction (Corrected to Lower Triangular)
        # ------------------------------------------------------------
        # V[t] = sum_{k=0}^t a^(t - k) * X[k]
        
        i = t[:, None] # (T, 1) -> Row: Output t
        j = t[None, :] # (1, T) -> Col: Input k
        
        # CORRECTED: i - j implies t - k
        diff = i - j   # (T, T)
        mask = diff >= 0 # Causal mask
        
        tau_matrix = torch.zeros((T, T), device=device, dtype=dtype)
        
        # Calculate powers a^(t-k)
        # broadcasting scalar 'a' is automatic
        tau_matrix = tau_matrix.masked_scatter(mask, (a ** diff[mask]).to(dtype))

        # coefficients for v_init (T,)
        tau_vec_init = a ** (t + 1.0)
        
        # coefficients for v_rest (T,)
        tau_vec_rest = 1.0 - tau_vec_init

        return tau_matrix, tau_vec_init, tau_vec_rest, tau

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1.0 + 1e-3)

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
    # ------------------------------------------------------------------
    # 多步前向传播 (适配 (T, B, C_sub, H, W, N))
    # ------------------------------------------------------------------
    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: Shape (T, B, C_sub, H, W, N)
        Returns:
            y: Shape (T, B, C_sub, H, W, N)
        """
        if self.no_filter == False:
            # 1. 形状获取
            T, B, C_sub, H, W, N = x_seq.shape
            device = x_seq.device
            dtype = x_seq.dtype

            (
                tau_matrix,   # (T, T)
                tau_vec_init, # (T,)
                tau_vec_rest, # (T,)
                tau,          # Scalar
            ) = self._build_tau_terms(T, device, dtype)

            # ------------------------------------------------------------
            # Step A: Decay Input Scaling
            # ------------------------------------------------------------
            if self.decay_input:
                x_seq = x_seq / tau # scalar division

            # ------------------------------------------------------------
            # Step B: Parallel Integration (Matrix Multiplication)
            # Goal: y[t] = M @ x
            # ------------------------------------------------------------
            
            # 1. Reshape x_seq to (T, -1) for standard matrix multiplication
            # We flatten (B, C_sub, H, W, N) into one dimension "Features"
            x_flat = x_seq.reshape(T, -1) 
            
            # 2. Perform Matmul
            # tau_matrix: (T, T)
            # x_flat:     (T, Features)
            # Result:     (T, Features)
            # Logic: out[t, f] = sum_k (M[t, k] * x[k, f])
            y1_flat = torch.matmul(tau_matrix, x_flat)
            
            # 3. Restore Shape
            # (T, B*C*H*W*N) -> (T, B, C_sub, H, W, N)
            y1 = y1_flat.view(T, B, C_sub, H, W, N)

            # ------------------------------------------------------------
            # Step C: Initial State Contribution (v_init)
            # ------------------------------------------------------------
            v_init = self.v
            if isinstance(v_init, torch.Tensor):
                v_init = v_init.detach()
                # v_init shape: (B, C_sub, H, W, N)
                # Add T dimension: (1, B, C_sub, H, W, N)
                v_init = v_init.unsqueeze(0)
            else:
                # scalar init
                v_init = torch.tensor(v_init, device=device, dtype=dtype)
                # scalar broadcasts automatically

            # tau_vec_init: (T,) -> (T, 1, 1, 1, 1, 1)
            tau_vec_init_broad = tau_vec_init.view(T, 1, 1, 1, 1, 1)
            
            y2 = tau_vec_init_broad * v_init

            # ------------------------------------------------------------
            # Step D: Resting Potential Contribution (v_rest)
            # ------------------------------------------------------------
            # tau_vec_rest: (T,) -> (T, 1, 1, 1, 1, 1)
            tau_vec_rest_broad = tau_vec_rest.view(T, 1, 1, 1, 1, 1)
            y3 = tau_vec_rest_broad * self.v_rest

            # ------------------------------------------------------------
            # Final Sum
            # ------------------------------------------------------------
            y = y1 + y2 + y3

            if self.store_v_seq:
                self.v_seq = y

            # Detach and store last state
            # State shape: (B, C_sub, H, W, N)
            self.v = y[-1].detach()
            if self.gating:
                gate_factor = torch.sigmoid(self.gate_scale * y + self.gate_beta)
                
                y_dendrite = y * gate_factor
            else:
                y_dendrite = y

            # ------------------------------------------------------------
            # 4. 直连融合
            # ------------------------------------------------------------
            # 输出 = 经过门控的低频积分 + 原始高频输入
            if self.res == True:
                y_out = y_dendrite + self.skip_weight * x_seq
            else:
                y_out = y_dendrite    
            
            return y_out
        
'''

class MultiScaleDendCompartment(BaseDendCompartment):
    """
    Multi-timescale Passive dendritic compartment with branch-wise learnable tau.
    
    Key Features:
    1. Independent tau for each branch N.
    2. Parallel integration using Causal (Lower Triangular) Matrix.
    3. Supports input shape (T, B, C_sub, H, W, N).
    """

    def __init__(
        self,
        num_branches: int,       
        init_tau = [1.25,5.0],  ###
        decay_input: bool = True, ###
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        gating = True,   #### 如果要加函数，就改这里
        bn = True
    ):
        super().__init__(v_rest, step_mode, store_v_seq)

        self.num_branches = num_branches
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.gating = gating
        self.bn = bn
        # ----------------------------------------------------------------
        # 1. 初始化 tau 为向量 (num_branches,)
        # ----------------------------------------------------------------
        if init_tau is None:
            # 默认线性分布初始化，模拟从快到慢 (2.0 ~ 10.0)
            tau_data = torch.linspace(2.0, 10.0, steps=num_branches)
        elif isinstance(init_tau, (float, int)):
            tau_data = torch.full((num_branches,), float(init_tau))
        else:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
            assert tau_data.shape[0] == num_branches, \
                f"init_tau shape {tau_data.shape} mismatch num_branches {num_branches}"

        self.tau = nn.Parameter(tau_data)
        if self.gating == True:
            
            if self.bn == True:
                self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches))
                self.branch_bn = nn.BatchNorm1d(num_branches, affine=True,momentum=0.1)
                nn.init.constant_(self.branch_bn.weight, 2.0)
                nn.init.constant_(self.branch_bn.bias, 0.5)
            else:
                self.gate_scale = nn.Parameter(torch.tensor(0.5))
                self.gate_beta2 = nn.Parameter(torch.tensor(0.0))    

    # ------------------------------------------------------------------
    # 2. 构建因果并行矩阵 (Lower Triangular Matrix)
    # ------------------------------------------------------------------
    def _build_tau_terms(self, T: int, device, dtype):
        """
        Builds tau-dependent matrices for parallel integration.
        
        Returns:
            tau_matrix: (N, T, T) -> Lower Triangular (Causal)
            tau_vec_init: (N, T)
            tau_vec_rest: (N, T)
            tau: (N,)
        """
        # 限制 tau 范围，防止除零
        tau = torch.clamp(self.tau, min=1.0 + 1e-3) # Shape: (N,)
        
        # a = 1 - 1/tau (LIF decay factor)
        a = 1.0 - 1.0 / tau # (N,)

        t = torch.arange(T, device=device, dtype=dtype) # (T,)

        # --------------------------------------------------------
        # Matrix Construction (Corrected to Lower Triangular)
        # --------------------------------------------------------
        # Target: V[t] = sum_{k=0}^{t} a^(t-k) * X[k]
        # Exponent = t_out - t_in
        
        i = t[:, None] # (T, 1) -> Output Time (Row)
        j = t[None, :] # (1, T) -> Input Time (Col)
        
        # CORRECTED: i - j implies t_out - t_in
        diff = i - j   # (T, T)
        
        # CORRECTED: mask for Lower Triangular (t_out >= t_in)
        mask = diff >= 0
        
        # Expand dims for N branches
        # diff: (T, T) -> (1, T, T)
        # a: (N,) -> (N, 1, 1)
        
        diff_expanded = diff.unsqueeze(0) 
        mask_expanded = mask.unsqueeze(0).expand(self.num_branches, T, T)
        a_expanded = a.view(self.num_branches, 1, 1)
        
        tau_matrix = torch.zeros((self.num_branches, T, T), device=device, dtype=dtype)
        
        # Broadcasting: a_expanded (N,1,1) ** diff_expanded (1,T,T) -> (N,T,T)
        pow_vals = a_expanded ** diff_expanded 
        
        # Fill Lower Triangular part
        tau_matrix = tau_matrix.masked_scatter(mask_expanded, (pow_vals[mask_expanded]).to(dtype))

        # coefficients for v_init (N, T)
        # V_init contrib at time t is: a^(t + 1) * v_init
        tau_vec_init = a_expanded.view(self.num_branches, 1) ** (t.view(1, T) + 1.0)
        
        # coefficients for v_rest (N, T)
        tau_vec_rest = 1.0 - tau_vec_init

        return tau_matrix, tau_vec_init, tau_vec_rest, tau

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1.0 + 1e-3)

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
    # ------------------------------------------------------------------
    # 3. 多步前向传播 (支持 N 在最后一维)
    # ------------------------------------------------------------------
    def multi_step_forward(self, x_seq: torch.Tensor):
        """
        Args:
            x_seq: Shape (T, B, C_sub, H, W, N)
        Returns:
            y: Shape (T, B, C_sub, H, W, N)
        """
        # 1. 形状检查
        T, B, C_sub, H, W, N = x_seq.shape
        assert N == self.num_branches, f"Mismatch branches: {N} vs {self.num_branches}"
        
        device = x_seq.device
        dtype = x_seq.dtype

        # 获取矩阵 (N, T, T)
        (
            tau_matrix,   # (N, T, T)
            tau_vec_init, # (N, T)
            tau_vec_rest, # (N, T)
            tau,          # (N,)
        ) = self._build_tau_terms(T, device, dtype)

        # ------------------------------------------------------------
        # Step A: Decay Input Scaling (广播机制，无开销)
        # ------------------------------------------------------------
        if self.decay_input:
            x_seq = x_seq / tau # (T, ..., N) / (N,) -> OK

        # ------------------------------------------------------------
        # Step B: Loop Integration (避免 Permute 的高性能写法)
        # ------------------------------------------------------------
        # 我们不把 N 换到前面，而是针对每个 N 独立做矩阵乘法
        # 因为 N 很小 (2, 4)，循环开销远小于 Tensor Copy 开销
        
        y_branches = []
        
        for n in range(self.num_branches):
            # 1. 取出第 n 个分支的数据 (切片操作，无 Copy)
            # x_n shape: (T, B, C_sub, H, W)
            x_n = x_seq[..., n] 
            
            # 2. Reshape 为 (T, -1) 以利用极速矩阵乘法
            x_n_flat = x_n.reshape(T, -1)
            
            # 3. 矩阵乘法: (T, T) @ (T, Features)
            # tau_matrix[n] shape is (T, T)
            y_n_flat = torch.matmul(tau_matrix[n], x_n_flat)
            
            # 4. 还原形状 (T, B, C_sub, H, W)
            y_n = y_n_flat.view(T, B, C_sub, H, W)
            
            # 5. 处理 Init 和 Rest (利用广播，开销极小)
            # tau_vec_init[n] shape: (T,) -> (T, 1, 1, 1, 1)
            t_init = tau_vec_init[n].view(T, 1, 1, 1, 1)
            t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1)
            
            # v_init 处理
            v_init = self.v
            if isinstance(v_init, torch.Tensor):
                # v_init: (B, C, H, W, N) -> v_init[..., n]: (B, C, H, W)
                v_in = v_init[..., n].detach().unsqueeze(0) # (1, B, ...)
            else:
                v_in = v_init # scalar
            
            # 累加
            y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
            
            y_branches.append(y_n)

        # 6. 堆叠回 (T, B, C, H, W, N)
        # stack 会发生一次内存拷贝，但这是无法避免的，且比之前的 permute 快得多
        y = torch.stack(y_branches, dim=-1)

        # ------------------------------------------------------------
        # Step C: Branch BN & Gating (保持不变)
        # ------------------------------------------------------------
        # 1. Flatten for BN
        if self.gating == True:
            if self.bn == True:
                y_flat = y.view(-1, N) 
                
                # 2. BN
                y_normed_flat = self.branch_bn(y_flat)
                
                # 3. Restore
                y_normed = y_normed_flat.view(T, B, C_sub, H, W, N)
                
                # 4. Gating

                gate = torch.sigmoid(self.gate_alpha * y_normed + self.gate_beta)
                y_out = y_normed * gate
            else:
                gate_factor = torch.sigmoid(self.gate_scale * y + self.gate_beta2)
                y_out = y * gate_factor    
        else:
            y_out = y
        # ------------------------------------------------------------
        # State Update
        # ------------------------------------------------------------
        if self.store_v_seq:
            self.v_seq = y 

        self.v = y[-1].detach()
        
        return y_out

'''

class MultiScaleDendCompartment(BaseDendCompartment):
    """
    Multi-timescale Passive dendritic compartment with Glia-Modulated Dynamics.
    
    Includes:
    1. Neuron: Multi-branch Integration (Fast/Slow branches).
    2. Oligodendrocyte: Activity-dependent dynamic Tau.
    3. Astrocyte: Context-dependent Gating / BN modulation.
    """

    def __init__(
        self,
        num_branches: int,
        c_sub=1,       
        init_tau = [1.25, 5.0],
        decay_input: bool = True,
        v_rest: float = 0.0,
        step_mode: str = "m",
        store_v_seq: bool = False,
        gating: bool = True,
        bn: bool = True,
        bn_alter:bool = True,
        use_astro: bool = True, 
        use_oli: bool = False
    ):
        super().__init__(v_rest, step_mode, store_v_seq)

        self.num_branches = num_branches
        self.c_sub = c_sub
        self.decay_input = decay_input
        self.v_rest = v_rest
        self.gating = gating
        self.bn = bn
        self.bn_alter = bn_alter
        self.use_astro = use_astro
        self.use_oli = use_oli

        # ----------------------------------------------------------------
        # 1. 基础神经元 Tau 初始化
        # ----------------------------------------------------------------
        if init_tau is None:
            tau_data = torch.linspace(2.0, 10.0, steps=num_branches)
        elif isinstance(init_tau, (float, int)):
            tau_data = torch.full((num_branches,), float(init_tau))
        else:
            tau_data = torch.as_tensor(init_tau, dtype=torch.float32)
        
        # 将 Tau 注册为 Parameter
        self.tau = nn.Parameter(tau_data)

        # ----------------------------------------------------------------
        # 2. 胶质细胞模块 (Glia Modules)
        # ----------------------------------------------------------------
        if self.use_oli:
            # [少突胶质细胞] 动态 Tau 调节系数
            # 髓鞘化灵敏度，初始化为 0.5
            self.oligo_alpha = nn.Parameter(torch.tensor(0.5))

        if self.use_astro:
            # [星形胶质细胞] 慢速积分状态
            # 钙离子衰减系数 (0.9 ~ 0.99)
            self.astro_decay = nn.Parameter(torch.tensor(0.95))
            # 钙离子对门控的影响力 (Gain)
            self.astro_gain = nn.Parameter(torch.tensor(1.0))
            # 内部状态
            self.ca_state = 0.0 

        # ----------------------------------------------------------------
        # 3. 激活函数与 BN 配置
        # ----------------------------------------------------------------
        if self.gating:
            if self.bn:

                # BN + Gating 模式 (静态图/DVS通用)
                self.gate_alpha = nn.Parameter(torch.ones(1, 1, 1, 1, 1, num_branches))
                self.gate_beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1, num_branches))
                if self.bn_alter==False:
                    self.branch_bn = nn.BatchNorm1d(num_branches, affine=True, momentum=0.1)
                else:
                    self.branch_bn = nn.BatchNorm2d(self.c_sub, affine=True, momentum=0.1) ##    
                nn.init.constant_(self.branch_bn.weight, 1.5)
                nn.init.constant_(self.branch_bn.bias, 0.5)
            else:
                # 纯 Gating 模式 (Swish-like, 更适合 DVS)
                self.gate_scale = nn.Parameter(torch.tensor(0.5))
                self.gate_beta2 = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------
    # 辅助: 动态 Tau 计算 (少突胶质细胞)
    # ------------------------------------------------------------------
    def _compute_dynamic_tau(self, x_seq):
        """
        根据输入活跃度调节 Tau。
        Activity High -> Tau Low (Myelination, Fast Conduction)
        """
        if not self.use_oli:
            return self.tau
        
        # 计算全局或通道级活跃度
        # x_seq: (T, B, C/N, H, W, N)
        # 取绝对值的均值作为能量指标
        T, B, C, H, W, N = x_seq.shape
        activity = x_seq.abs().mean(dim=(0, 1, 2, 3, 4)) # (N,)
        
        # 广播到 Tau 的形状 (N,) -> (B, N)
        #base_tau = self.tau.view(B, N)
        
        # 动态公式: Tau_eff = Base / (1 + alpha * activity)
        # 限制最小 Tau 为 1.05 (防止除以 0 或过小)
        tau_eff = self.tau / (1.0 + self.oligo_alpha * activity)
        tau_eff = torch.clamp(tau_eff, min=1.05)
        
        return tau_eff

    # ------------------------------------------------------------------
    # 辅助: 动态门控调节 (星形胶质细胞)
    # ------------------------------------------------------------------
    def _compute_astro_modulation(self, y_seq):
        """
        计算星形胶质细胞的钙信号，用于调节 BN/Gating 的强度。
        """
        if not self.use_astro:
            return 1.0 # 无调节
            
        T = y_seq.shape[0]
        B = y_seq.shape[1]
        device = y_seq.device
        
        # 1. 能量输入 (取绝对值)
        energy_seq = y_seq.abs().mean(dim=(2, 3, 4, 5)) # (T, B) - 简化为全图能量
        
        # 2. 慢速积分 (Leaky Integration)
        ca_seq = []
        
        curr_ca = torch.zeros(B, device=device)

        decay = torch.clamp(self.astro_decay, 0.0, 1.0)
        
        for t in range(T):
            curr_ca = decay * curr_ca + (1 - decay) * energy_seq[t]
            ca_seq.append(curr_ca)
            
        # 更新状态
        self.ca_state = curr_ca.detach()
        
        # 3. 生成调节系数 (Modulation Factor)
        ca_signal = torch.stack(ca_seq, dim=0).view(T, -1, 1, 1, 1, 1)
        modulation = torch.sigmoid(self.astro_gain * ca_signal) 
        
        return modulation
    # ------------------------------------------------------------------
    # 构建矩阵 (Loop Mode for Dynamic Tau or Matrix Mode for Static)
    # ------------------------------------------------------------------
    # 为了兼容动态 Tau，这里建议使用 Loop 模式积分，或者近似为 Matrix
    # 这里为了保持你原代码的 Matrix 结构，我们假设 Tau 在 Batch 内取平均
    # 或者如果 use_glia=False，则回退到原逻辑
    
    def _build_tau_terms(self, T, device, dtype, effective_tau=None):
        # 如果传入了动态 Tau (B, N)，这里取均值做近似，或者拓展维度
        # 为了矩阵构建方便，我们取 Batch 均值 (近似法)
        if effective_tau is not None:
            tau_val = effective_tau # (N,)
        else:
            tau_val = self.tau
            
        tau = torch.clamp(tau_val, min=1.0 + 1e-3)
        a = 1.0 - 1.0 / tau
        t = torch.arange(T, device=device, dtype=dtype)
        
        # ... 原矩阵构建代码 (保持不变) ...
        i = t[:, None]; j = t[None, :]; diff = i - j; mask = diff >= 0
        
        diff_expanded = diff.unsqueeze(0) 
        mask_expanded = mask.unsqueeze(0).expand(self.num_branches, T, T)
        a_expanded = a.view(self.num_branches, 1, 1)
        
        tau_matrix = torch.zeros((self.num_branches, T, T), device=device, dtype=dtype)
        pow_vals = a_expanded ** diff_expanded
        tau_matrix = tau_matrix.masked_scatter(mask_expanded, pow_vals[mask_expanded].to(dtype))
        
        tau_vec_init = a_expanded.view(self.num_branches, 1) ** (t.view(1, T) + 1.0)
        tau_vec_rest = 1.0 - tau_vec_init
        
        return tau_matrix, tau_vec_init, tau_vec_rest, tau
    
    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1.0 + 1e-3)

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

    # ------------------------------------------------------------------
    # 多步前向传播
    # ------------------------------------------------------------------
    def multi_step_forward(self, x_seq: torch.Tensor):
        T, B, C_sub, H, W, N = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype
        self.c_sub = C_sub  ##
        # --------------------------------------
        # Step 1: 少突胶质细胞 - 动态 Tau
        # --------------------------------------
        if self.use_oli:
            tau_eff = self._compute_dynamic_tau(x_seq)
        else:
            tau_eff = None

        # 构建积分矩阵
        (tau_matrix, tau_vec_init, tau_vec_rest, tau_scalar) = \
            self._build_tau_terms(T, device, dtype, effective_tau=tau_eff)

        # --------------------------------------
        # Step 2: 输入缩放 (Decay Input)
        # --------------------------------------
        if self.decay_input:
            # 如果是动态 Tau，用动态值除；否则用标量除
            div_tau = tau_eff.view(1, 1, 1, 1, 1, N) if tau_eff is not None else tau_scalar.view(1, 1, 1, 1, 1, N)
            x_seq = x_seq / div_tau

        # --------------------------------------
        # Step 3: 矩阵积分 (Matrix Integration)
        # --------------------------------------
        y_branches = []
        for n in range(self.num_branches):
            x_n = x_seq[..., n].reshape(T, -1)
            y_n_flat = torch.matmul(tau_matrix[n], x_n)
            y_n = y_n_flat.view(T, B, C_sub, H, W)
            
            # Init & Rest
            t_init = tau_vec_init[n].view(T, 1, 1, 1, 1)
            t_rest = tau_vec_rest[n].view(T, 1, 1, 1, 1)
            
            v_init = self.v
            if isinstance(v_init, torch.Tensor):
                v_in = v_init[..., n].detach().unsqueeze(0)
            else:
                v_in = v_init
                
            y_n = y_n + (t_init * v_in) + (t_rest * self.v_rest)
            y_branches.append(y_n)

        y = torch.stack(y_branches, dim=-1) # (T, B, ..., N)
        
        # 更新状态
        if self.store_v_seq: self.v_seq = y 
        self.v = y[-1].detach()

        # --------------------------------------
        # Step 4: 激活与 BN (集成星形胶质细胞)
        # --------------------------------------
        if self.gating:
            # 计算星形胶质细胞调节系数 (Modulation)
            # modulation: (T, B, 1, 1, 1, 1) or 1.0
            astro_mod = self._compute_astro_modulation(y)

            if self.bn:
                # [模式 A: BN + Sigmoid Gate]
                # 这是最强大的组合：BN 提供标准化，Gating 提供非线性，Astro 提供上下文
                if self.bn_alter==False:
                    y_flat = y.view(-1, N)
                    y_normed_flat = self.branch_bn(y_flat)
                    y_normed = y_normed_flat.view(T, B, C_sub, H, W, N)
                else:
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
                
                # 应用胶质调节：
                # 如果 Astro 认为这里是噪声(mod~0)，则压制 BN 的输出
                y_normed = y_normed * astro_mod 
                
                # 最后的非线性门控
                gate = torch.sigmoid(self.gate_alpha * y_normed + self.gate_beta)
                y_out = y_normed * gate
                
            else:
                # [模式 B: Swish-like Gating]
                # y_out = y * Sigmoid(scale * y + beta)
                # 同样可以加上胶质调节
                
                y = y * astro_mod # 先调节幅度
                
                gate_factor = torch.sigmoid(self.gate_scale * y + self.gate_beta2)
                y_out = y * gate_factor    
        else:
            y_out = y
            
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
