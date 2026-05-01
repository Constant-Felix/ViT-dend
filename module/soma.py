from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate


class BaseSoma(neuron.BaseNode):

    def __init__(
        self, v_threshold: float = 1., v_reset: float = 0., 
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode: str = "s", 
        backend: str = "torch", 
        store_v_seq: bool = False, store_v_pre_spike: bool = False
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode,
            backend, store_v_seq
        )
        self.store_v_pre_spike = store_v_pre_spike

    @property
    def store_v_pre_spike(self) -> bool:
        return self._store_v_pre_spike

    @store_v_pre_spike.setter
    def store_v_pre_spike(self, val: bool):
        self._store_v_pre_spike = val
        if val and (not hasattr(self, "v_pre_spike")):
            self.register_memory("v_pre_spike", None)

    def _single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        v_pre_spike = self.v
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike, v_pre_spike

    def single_step_forward(self, x: torch.Tensor):
        spike, v_pre_spike = self._single_step_forward(x)
        if self.store_v_pre_spike:
            self.v_pre_spike = v_pre_spike
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_pre_spike:
            v_pre_spike_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y, v_pre_spike = self._single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_pre_spike:
                v_pre_spike_seq.append(v_pre_spike)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        if self.store_v_pre_spike:
            self.v_pre_spike = torch.stack(v_pre_spike_seq)
        return torch.stack(y_seq)

def expand_tensor_cumulative(tensor, max_value=4):

    T, B, C, H, W = tensor.shape
    # 创建一个 shape 为 [max_value, 1, 1, 1, 1, 1] 的比较向量
    steps = torch.arange(max_value, device=tensor.device).view(max_value, 1, 1, 1, 1, 1)

    # 扩展原始张量维度，便于比较 → [1, T, B, C, H, W]
    tensor_expanded = tensor.unsqueeze(0)

    # 比较：每个位置 v，生成 v 个 1，其余为 0
    binary = (steps < tensor_expanded).float()  # → shape [max_value, T, B, C, H, W]

    # 重新 reshape → [max_value * T, B, C, H, W]
    binary = binary.permute(1, 0, 2, 3, 4, 5).reshape(T * max_value, B, C, H, W)

    return binary

class MultiSpike8(nn.Module):

    class quant8(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.floor(torch.clamp(input, min=0, max=8))   ####改成了取整

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
        return self.quant8.apply(x) 


class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)
     

class IntergerSoma(neuron.BaseNode):
    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0.,detach_reset: bool = False,decay_input: bool = True, ##
        step_mode='m', backend='torch', thre = 4,
        surrogate_function: Callable = surrogate.Sigmoid()
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function,detach_reset,step_mode,backend
        )
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.decay_input = decay_input
        if thre == 4:
            self.qtrick = MultiSpike4()
        if thre == 8:
            self.qtrick = MultiSpike8()   
    
    def multi_step_forward(self, x: torch.Tensor):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        tau = torch.clamp(self.tau, min=1+1e-3)
        for i in range(time_window):
            if i >= 1:
                if self.decay_input == False:
                    mem = (mem_old - spike.detach())/tau + x[i]
                else:
                    mem = (mem_old - spike.detach())/tau + x[i]*(1-1/tau)
            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        #h = hook    
        return output


class IntergerSoma_infer(neuron.BaseNode):   ###暂时我没有改这里的tau
    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0.,detach_reset: bool = False,
        step_mode='m', backend='torch', 
        surrogate_function: Callable = surrogate.Sigmoid()
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function,detach_reset,step_mode,backend
        )
        self.tau = tau
        self.qtrick = MultiSpike4()
    
    def multi_step_forward(self, x: torch.Tensor):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) / self.tau + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike

        spike = expand_tensor_cumulative(output)

        return spike
    
import torch
import torch.nn as nn

class SSF_Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, U, v_th):
        ctx.save_for_backward(input)
        ctx.U = U
        # 根据论文公式(17)：先截断，除以阈值，最后向下取整 (floor)
        clipped_input = torch.clamp(input, min=-U, max=U)
        return torch.floor(clipped_input / v_th)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        U = ctx.U
        grad_input = grad_output.clone()
        
        # 替代梯度 (STE): 在有效截断区间 [-U, U] 内放行梯度，超出则截断为 0
        grad_input[input < -U] = 0
        grad_input[input > U] = 0
        return grad_input, None, None  # 对应 input, U, v_th 的梯度，常数不需要梯度

class SSF(nn.Module):
    def __init__(self, U: int = 4, v_th: float = 1.0):
        super().__init__()
        self.U = U
        self.v_th = v_th
        
        # 【核心新增】引入可学习的平移参数 phi_p 和缩放参数 phi_s
        # 初始化为不改变原分布的状态 (phi_p=0, phi_s=1)
        self.phi_p = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.phi_s = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        # 1. 膜电位平移与缩放 (PyTorch 的 autograd 会自动计算 phi_p 和 phi_s 的梯度)
        # 为防止缩放因子在训练中更新至负数或极小值导致除零错误，进行极小值限制
        phi_s_clamped = torch.clamp(self.phi_s, min=1e-3)
        x_norm = (x - self.phi_p) / phi_s_clamped
        
        # 2. 离散化与激活 (进入自定义的直通估计器)
        spike = SSF_Quant.apply(x_norm, self.U, self.v_th)
        return spike

class IntergerSoma_ssf(neuron.BaseNode): # 假设基于 SpikingJelly 或类似框架的基类
    def __init__(
        self, tau: float = 2.,
        v_threshold: float = 1., v_reset: float = 0., detach_reset: bool = False, decay_input: bool = True,
        step_mode='m', backend='torch', thre = 4,
        surrogate_function: Callable = surrogate.Sigmoid()
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend
        )
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.decay_input = decay_input
        
        # 替换原有的 MultiSpike 为 SSF 机制
        self.qtrick = SSF(U=thre, v_th=v_threshold)
    
    def multi_step_forward(self, x: torch.Tensor):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        tau = torch.clamp(self.tau, min=1+1e-3)
        
        for i in range(time_window):
            if i >= 1:
                if self.decay_input == False:
                    mem = (mem_old)/tau + x[i]
                else:
                    mem = (mem_old)/tau + x[i]*(1-1/tau)
            else:
                mem = x[i]
                
            # 这里输入给 SSF 的是原始膜电位，SSF 内部会自动进行平移和缩放
            spike = self.qtrick(mem)

            mem_old = mem.clone()
            output[i] = spike
            
        return output

class LIFSoma(BaseSoma):

    def __init__(
        self, tau: float = 2., decay_input: bool = True, 
        v_threshold: float = 1., v_reset: float = 0., 
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode: str = "s", 
        backend: str = "torch", 
        store_v_seq: bool = False, store_v_pre_spike: bool = False
    ):
        if not (isinstance(tau, float) and tau >= 1.):
            return AssertionError(
                f"LIFSoma.tau should be larger than 1., "
                f"but get tau={tau}."
            )
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, 
            step_mode, backend, store_v_seq, store_v_pre_spike
        )
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.decay_input = decay_input

    def extra_repr(self):
        return (
            super().extra_repr() + 
            f", tau={self.tau}, decay_input={self.decay_input}"
        )

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_decay_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_decay(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_no_decay_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v  - v / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_neuronal_charge_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        v_pre_spike = v + (x - (v - v_reset)) / tau
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = spike * v_reset + (1. - spike) * v_pre_spike
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        v_pre_spike = v - (v - v_reset) / tau + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = spike * v_reset + (1. - spike) * v_pre_spike
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v_pre_spike = v + (x - v) / tau
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft_no_decay(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v_pre_spike = v - v / tau + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_no_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_no_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, 
        v_reset: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_no_decay(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_no_decay_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    def neuronal_charge(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        if self.decay_input:
            if (self.v_reset is None) or (self.v_reset == 0.):
                self.v = self.jit_neuronal_charge_decay_reset0(
                    x, self.v, tau
                )
            else:
                self.v = self.jit_neuronal_charge_decay(
                    x, self.v, self.v_reset, tau
                )
        else:
            if (self.v_reset is None) or (self.v_reset == 0.):
                self.v = self.jit_neuronal_charge_no_decay_reset0(
                    x, self.v, tau
                )
            else:
                self.v = self.jit_neuronal_charge_no_decay(
                    x, self.v, self.v_reset, tau
                )

    def single_step_forward(self, x: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None: # soft reset
                if self.decay_input:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_soft_decay(
                            x, self.v, self.v_threshold, tau
                        )
                    )
                else:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_soft_no_decay(
                            x, self.v, self.v_threshold, tau
                        )
                    )
            else:
                if self.decay_input:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_hard_decay(
                            x, self.v, self.v_threshold, self.v_reset, tau
                        )
                    )
                else:
                    spike, v_pre_spike, self.v = (
                        self.jit_single_eval_hard_no_decay(
                            x, self.v, self.v_threshold, self.v_reset, tau
                        )
                    )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        tau = torch.clamp(self.tau, min=1+1e-3)
        if self.training:
            return super().multi_step_forward(x_seq)
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None: # soft reset
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_soft_decay_v_seq(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_soft_decay(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_soft_no_decay_v_seq(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_soft_no_decay(
                                x_seq, self.v, self.v_threshold, tau
                            )
                        )
            else: # hard reset
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_hard_decay_v_seq(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_hard_decay(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
                else:
                    if self.store_v_seq:
                        spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                            self.jit_multi_eval_hard_no_decay_v_seq(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
                    else:
                        spike_seq, v_pre_spike_seq, self.v = (
                            self.jit_multi_eval_hard_no_decay(
                                x_seq, self.v, self.v_threshold, 
                                self.v_reset, tau
                            )
                        )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike_seq
            return spike_seq


class IFSoma(BaseSoma):

    def __init__(
        self, v_threshold: float = 1., v_reset: float = 0.,
        surrogate_function: Callable = surrogate.Sigmoid(), 
        detach_reset: bool = False, step_mode='s', backend='torch', 
        store_v_seq: bool = False, store_v_pre_spike: bool = False
    ):
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, 
            step_mode, backend, store_v_seq, store_v_pre_spike
        )

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    @torch.jit.script
    def jit_single_eval_hard(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        v_pre_spike = v + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_reset * spike + v_pre_spike * (1. - spike)
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_single_eval_soft(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        v_pre_spike = v + x
        spike = (v_pre_spike >= v_threshold).to(x)
        v_post_spike = v_pre_spike - spike * v_threshold
        return spike, v_pre_spike, v_post_spike

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_hard_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v_pre_spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_multi_eval_soft_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_pre_spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            v_pre_spike_seq[t] = v
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v_pre_spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None: # soft reset
                spike, v_pre_spike, self.v = (
                    self.jit_single_eval_soft(x, self.v, self.v_threshold)
                )
            else:
                spike, v_pre_spike, self.v = (
                    self.jit_single_eval_hard(
                        x, self.v, self.v_threshold, self.v_reset
                    )
                )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            return super().multi_step_forward(x_seq)
        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None: # soft reset
                if self.store_v_seq:
                    spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                        self.jit_multi_eval_soft_v_seq(
                            x_seq, self.v, self.v_threshold
                        )
                    )
                else:
                    spike_seq, v_pre_spike_seq, self.v = (
                        self.jit_multi_eval_soft(
                            x_seq, self.v, self.v_threshold
                        )
                    )
            else:
                if self.store_v_seq:
                    spike_seq, v_pre_spike_seq, self.v, self.v_seq = (
                        self.jit_multi_eval_hard_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset
                        )
                    )
                else:
                    spike_seq, v_pre_spike_seq, self.v = (
                        self.jit_multi_eval_hard(
                            x_seq, self.v, self.v_threshold, self.v_reset
                        )
                    )
            if self.store_v_pre_spike:
                self.v_pre_spike = v_pre_spike_seq
            return spike_seq