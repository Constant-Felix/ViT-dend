import torch
import torchinfo
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import dendrite,dend_compartment,soma,neuron
from module import wiring

class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_dend_Conv_int(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
        dend = False,
        integer = False,
        num_compartment = 4
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features*num_compartment)
        self.res = in_features == hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_compartment = num_compartment
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.f_da = lambda x:x
        if dend == False:
            if spike_mode == "lif":
                self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.fc1_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
        else:
            self.dc1 = dend_compartment.PassiveDendCompartment(tau=2, decay_input=True, step_mode="m")
            self.wr1 = wiring.SegregatedDendWiring(num_compartment)
            self.dend1 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc1,wiring=self.wr1)
            if integer == False:
                self.soma1 = soma.LIFSoma(step_mode='m')
            else:
                self.soma1 = soma.IntergerSoma_infer(step_mode='m')    
            self.fc1_lif = neuron.VActivationForwardDendNeuron(dend=self.dend1,soma=self.soma1,f_da=self.f_da,soma_shape=np.zeros((3,),int))
                
        
        self.fc2_conv = nn.Conv2d(
            in_features, hidden_features, kernel_size=1, stride=1
        )
        #print(self.fc2_conv)
        self.fc2_bn = nn.BatchNorm2d(hidden_features)
        if dend == False:
            if spike_mode == "lif":
                self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.fc2_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
        else:
            self.dc2 = dend_compartment.PassiveDendCompartment(tau=2, decay_input=True, step_mode="m")
            self.wr2 = wiring.SegregatedDendWiring(num_compartment)
            self.dend2 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc2,wiring=self.wr2)
            if integer == False:
                self.soma2 = soma.LIFSoma(step_mode='m')
            else:
                self.soma2 = soma.IntergerSoma_infer(step_mode='m')
            self.fc2_lif = neuron.VActivationForwardDendNeuron(dend=self.dend2,soma=self.soma2,f_da=self.f_da,soma_shape=np.zeros((3,),int))        
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):  ###需验证这里形状
        T, B, C, H, W = x.shape
        identity = x

        self.fc1_lif.soma_shape[1] = x.shape[3]  ######
        self.fc1_lif.soma_shape[2] = x.shape[4]
        self.fc1_lif.soma_shape[0] = int(x.shape[2]/self.num_compartment)
        x = self.fc1_lif(x)
        #id = x
        #print(f"shape is {x.shape}")
        #new_shape = (4, x.shape[0] // 4) + x.shape[1:]
        #x = x.view(new_shape).sum(dim=1)
        #print(f"shape is {x.shape}")
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        shape = (4*T,B) + x.shape[1:]
        x = x.view(shape)
        new_shape = (T, x.shape[0] // T) + x.shape[1:]
        x = x.view(new_shape).sum(dim=1)
        x = x.view(T*B,-1,H,W)   
        x = self.fc1_bn(x).reshape(T, B, -1, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        #x = self.sample_up_1(x.flatten(0,1)).reshape(T,B,-1,H,W)    
        self.fc2_lif.soma_shape[1] = x.shape[3]  ######
        self.fc2_lif.soma_shape[2] = x.shape[4]
        self.fc2_lif.soma_shape[0] = int(x.shape[2]/self.num_compartment)   
        x = self.fc2_lif(x)
        #new_shape = (4, x.shape[0] // 4) + x.shape[1:]
        #x = x.view(new_shape).sum(dim=1)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        shape = (4*T,B) + x.shape[1:]
        x = x.view(shape)
        new_shape = (T, x.shape[0] // T) + x.shape[1:]
        x = x.view(new_shape).sum(dim=1)
        x = x.view(T*B,-1,H,W) 
        #print(f"shape is {x.shape}")
        x = self.fc2_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = x + identity
        #print(f"this is layer {self.layer}")
        return x, hook


class MS_SSA_dend_Conv_int(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        num_compartment=4,
        dend = False,  ###
        integer = False,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.f_da = lambda x:x
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend == False:
            if spike_mode == "lif":
                self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.q_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
        else:
            self.dc1 = dend_compartment.PassiveDendCompartment(tau=2, decay_input=True, step_mode="m")
            self.wr1 = wiring.SegregatedDendWiring(num_compartment)
            self.dend1 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc1,wiring=self.wr1)
            if integer == False:
                self.soma1 = soma.LIFSoma(step_mode='m')
            else:
                self.soma1 = soma.IntergerSoma(step_mode='m')
            self.q_lif = neuron.VActivationForwardDendNeuron(dend=self.dend1,soma=self.soma1,f_da=self.f_da,soma_shape=np.zeros((3,),int))         

        self.k_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend == False:
            if spike_mode == "lif":
                self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.k_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
        else:
            self.dc2 = dend_compartment.PassiveDendCompartment(tau=2, decay_input=True, step_mode="m")
            self.wr2 = wiring.SegregatedDendWiring(num_compartment)
            self.dend2 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc2,wiring=self.wr2)
            if integer == False:
                self.soma2 = soma.LIFSoma(step_mode='m')
            else:
                self.soma2 = soma.IntergerSoma(step_mode='m')
            self.k_lif = neuron.VActivationForwardDendNeuron(dend=self.dend2,soma=self.soma2,f_da=self.f_da,soma_shape=np.zeros((3,),int))


        self.v_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend == False:
            if spike_mode == "lif":
                self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
            elif spike_mode == "plif":
                self.v_lif = MultiStepParametricLIFNode(
                    init_tau=2.0, detach_reset=True, backend="cupy"
                )
        else:
            self.dc3 = dend_compartment.PassiveDendCompartment(tau=2, decay_input=True, step_mode="m")
            self.wr3 = wiring.SegregatedDendWiring(num_compartment)
            self.dend3 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc3,wiring=self.wr3)
            if integer == False:
                self.soma3 = soma.LIFSoma(step_mode='m')
            else:
                self.soma3 = soma.IntergerSoma(step_mode='m')
            self.v_lif = neuron.VActivationForwardDendNeuron(dend=self.dend3,soma=self.soma3,f_da=self.f_da,soma_shape=np.zeros((3,),int))        

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        #self.talking_heads = nn.Conv1d(
            #num_heads, num_heads, kernel_size=1, stride=1, bias=False
        #)
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        if dend == False:
            self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.proj_bn = nn.BatchNorm2d(dim)
        else:
            self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1) ##
            self.proj_bn = nn.BatchNorm2d(dim)    ##

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.num_compartment = num_compartment
        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, -1, H, W).contiguous()
        self.q_lif.soma_shape[0] = q_conv_out.shape[2]/self.num_compartment
        self.q_lif.soma_shape[1] = q_conv_out.shape[3]
        self.q_lif.soma_shape[2] = q_conv_out.shape[4]

        q_conv_out = self.q_lif(q_conv_out)  ###
        #new_shape = (4, q_conv_out.shape[0] // 4) + q_conv_out.shape[1:]
        #q_conv_out = q_conv_out.view(4*T,B,-1,H,W)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, -1, H, W).contiguous()
        self.k_lif.soma_shape[0] = k_conv_out.shape[2]/self.num_compartment
        self.k_lif.soma_shape[1] = k_conv_out.shape[3]
        self.k_lif.soma_shape[2] = k_conv_out.shape[4]

        k_conv_out = self.k_lif(k_conv_out)
        #new_shape = (4, k_conv_out.shape[0] // 4) + k_conv_out.shape[1:]
        #k_conv_out = k_conv_out.view(4*T,B,-1,H,W)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, -1, H, W).contiguous()
        self.v_lif.soma_shape[0] = v_conv_out.shape[2]/self.num_compartment
        self.v_lif.soma_shape[1] = v_conv_out.shape[3]
        self.v_lif.soma_shape[2] = v_conv_out.shape[4]

        v_conv_out = self.v_lif(v_conv_out)
        #new_shape = (4, v_conv_out.shape[0] // 4) + v_conv_out.shape[1:]
        #v_conv_out = v_conv_out.view(4*T,B,-1,H,W)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B h N C//h

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v, hook


class MS_Block_dend_Conv_int(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        num_compartment = 4,
        dend = False,
        integer = False
    ):
        super().__init__()
        self.attn = MS_SSA_dend_Conv_int(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
            dend=dend,
            num_compartment=num_compartment,
            integer=integer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.num_compartment = num_compartment
        #mlp_hidden_dim = int(dim * num_compartment)
        if dim % num_compartment != 0:
            print(f"dim should be divided by num compartment but got dim {dim} and n_com {num_compartment} instead")
        self.mlp = MS_MLP_dend_Conv_int(
            in_features=int(dim/num_compartment),
            hidden_features=dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
            num_compartment=num_compartment,
            dend=dend,
            integer=integer
        )
        #print(f"in feature: {self.mlp.in_features}, hidden: {self.mlp.hidden_features}")

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook