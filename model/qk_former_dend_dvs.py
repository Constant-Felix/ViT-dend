import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
from module import dendrite,dend_compartment,soma,neuron,wiring

__all__ = ['QKFormer_dend_dvs']


def _make_dend_compartment(c_sub, num_compartment, multi=False, soma_astro=False):
    if soma_astro:
        return dend_compartment.PureMultiScaleDendCompartment(
            num_compartment, step_mode='m', c_sub=c_sub
        )
    if multi == False:
        return dend_compartment.PassiveDendCompartment(step_mode="m", c_sub=c_sub)
    return dend_compartment.AdvancedNGCUDendCompartment(
        num_compartment, step_mode='m', c_sub=c_sub
    )


def _make_soma(integer=False, soma_astro=False):
    if soma_astro:
        return soma.AstroIntergerSoma(step_mode='m') if integer else soma.AstroLIFSoma(step_mode='m')
    return soma.IntergerSoma(step_mode='m') if integer else soma.LIFSoma(step_mode='m')


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,
                 dend = False,integer = False,multi = False,num_compartment=2,bn_alter=False,soma_astro=False):
        super().__init__()
        self.f_da = lambda x:x
        out_features = out_features or in_features
        #hidden_features = int(in_features*num_compartment)
        self.bn_alter = bn_alter
        self.in_features = in_features
        self.num_compartment = num_compartment
        #hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        if dend==False:
            self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc1 = _make_dend_compartment(hidden_features, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr1 = wiring.SegregatedDendWiring(num_compartment)
            self.dend1 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc1,wiring=self.wr1)
            self.soma1 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.mlp1_lif = neuron.VActivationForwardDendNeuron(dend=self.dend1,soma=self.soma1,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)
            #self.mlp1_lif.forward_strength.data = torch.full((num_compartment,),1.0)    

        self.mlp2_conv = nn.Conv2d(in_features, num_compartment*hidden_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(num_compartment*hidden_features)
        if dend==False:
            self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc2 = _make_dend_compartment(hidden_features*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr2 = wiring.SegregatedDendWiring(num_compartment)
            self.dend2 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc2,wiring=self.wr2)
            self.soma2 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.mlp2_lif = neuron.VActivationForwardDendNeuron(dend=self.dend2,soma=self.soma2,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)
            #self.mlp2_lif.forward_strength.data = torch.full((num_compartment,),1.0)    

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        self.mlp1_lif.soma_shape[1] = x.shape[3]  ######
        self.mlp1_lif.soma_shape[2] = x.shape[4]
        self.mlp1_lif.soma_shape[0] = int(x.shape[2]/self.num_compartment)
        #if self.bn_alter == True:
            #self.mlp1_lif.dend.compartment.c_sub = x.shape[2]
        x,_ = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, self.c_hidden*self.num_compartment, H, W)
        self.mlp2_lif.soma_shape[1] = x.shape[3]  ######
        self.mlp2_lif.soma_shape[2] = x.shape[4]
        self.mlp2_lif.soma_shape[0] = int(x.shape[2]/self.num_compartment)
        #if self.bn_alter == True:
            #self.mlp2_lif.dend.compartment.c_sub = x.shape[2]
        x,_ = self.mlp2_lif(x)

        return x

class Token_dend_QK_Attention(nn.Module):   ###改成conv2d
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,dend=False,integer=False,
                 num_compartment=2,multi=False,bn_alter=False,soma_astro=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_compartment = num_compartment
        self.bn_alter = bn_alter
        self.f_da = lambda x:x
        self.q_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim*num_compartment)

        if dend==False:
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc1 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr1 = wiring.SegregatedDendWiring(num_compartment)
            self.dend1 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc1,wiring=self.wr1)
            self.soma1 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.q_lif = neuron.VActivationForwardDendNeuron(dend=self.dend1,soma=self.soma1,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)         
            #self.q_lif.forward_strength.data = torch.full((num_compartment,),1.0)

        self.k_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend==False:
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc2 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr2 = wiring.SegregatedDendWiring(num_compartment)
            self.dend2 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc2,wiring=self.wr2)
            self.soma2 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.k_lif = neuron.VActivationForwardDendNeuron(dend=self.dend2,soma=self.soma2,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)
            #self.k_lif.forward_strength.data = torch.full((num_compartment,),1.0)    

        if integer==False:
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)
        else:
            self.attn_lif = soma.IntergerSoma_ssf()    

        self.proj_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend==False:
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc3 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr3 = wiring.SegregatedDendWiring(num_compartment)
            self.dend3 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc3,wiring=self.wr3)
            self.soma3 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.proj_lif = neuron.VActivationForwardDendNeuron(dend=self.dend3,soma=self.soma3,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)
            #self.proj_lif.forward_strength.data = torch.full((num_compartment,),1.0)


    def forward(self, x):
        T, B, C, H, W = x.shape

        #x = x.flatten(3)
        #T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, -1, H, W)

        self.q_lif.soma_shape[0] = q_conv_out.shape[2]//self.num_compartment
        self.q_lif.soma_shape[1] = q_conv_out.shape[3]
        self.q_lif.soma_shape[2] = q_conv_out.shape[4]
        #if self.bn_alter == True:
            #self.q_lif.dend.compartment.c_sub = q_conv_out.shape[2]
        q_conv_out,_ = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, H, W)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, -1, H, W)

        self.k_lif.soma_shape[0] = k_conv_out.shape[2]/self.num_compartment
        self.k_lif.soma_shape[1] = k_conv_out.shape[3]
        self.k_lif.soma_shape[2] = k_conv_out.shape[4]
        #if self.bn_alter == True:
            #self.k_lif.dend.compartment.c_sub = k_conv_out.shape[2]
        k_conv_out,_ = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, H, W)

        q = torch.sum(q, dim = 3, keepdim = True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, -1, H, W)

        #if self.bn_alter == True:
            #self.proj_lif.dend.compartment.c_sub = x.shape[2]
        self.proj_lif.soma_shape[0] = x.shape[2]/self.num_compartment
        self.proj_lif.soma_shape[1] = x.shape[3]
        self.proj_lif.soma_shape[2] = x.shape[4]
        
        x,_ = self.proj_lif(x)

        return x

class Spiking_dend_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 dend=False,integer=False,num_compartment=2,multi=False,bn_alter=False,soma_astro=False):
        
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.f_da = lambda x:x
        self.num_compartment = num_compartment
        self.bn_alter = bn_alter
        head_dim = dim // num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend==False:
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc1 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr1 = wiring.SegregatedDendWiring(num_compartment)
            self.dend1 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc1,wiring=self.wr1)
            self.soma1 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.q_lif = neuron.VActivationForwardDendNeuron(dend=self.dend1,soma=self.soma1,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)         
            #self.q_lif.forward_strength.data = torch.full((num_compartment,),1.0)   

        self.k_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend==False:
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc2 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr2 = wiring.SegregatedDendWiring(num_compartment)
            self.dend2 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc2,wiring=self.wr2)
            self.soma2 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.k_lif = neuron.VActivationForwardDendNeuron(dend=self.dend2,soma=self.soma2,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)
            #self.k_lif.forward_strength.data = torch.full((num_compartment,),1.0)     

        self.v_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend==False:
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc3 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr3 = wiring.SegregatedDendWiring(num_compartment)
            self.dend3 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc3,wiring=self.wr3)
            self.soma3 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.v_lif = neuron.VActivationForwardDendNeuron(dend=self.dend3,soma=self.soma3,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)        
            #self.v_lif.forward_strength.data = torch.full((num_compartment,),1.0)    

        if integer==False:
            self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)
        else:
            self.attn_lif = soma.IntergerSoma_ssf()    

        self.proj_conv = nn.Conv2d(dim, dim*num_compartment, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim*num_compartment)
        if dend==False:
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.dc4 = _make_dend_compartment(dim*num_compartment, num_compartment, multi=multi, soma_astro=soma_astro)
            self.wr4 = wiring.SegregatedDendWiring(num_compartment)
            self.dend4 = dendrite.SegregatedDend(step_mode='m',compartment=self.dc4,wiring=self.wr4)
            self.soma4 = _make_soma(integer=integer, soma_astro=soma_astro)
            self.proj_lif = neuron.VActivationForwardDendNeuron(dend=self.dend4,soma=self.soma4,f_da=self.f_da,soma_shape=np.zeros((3,),int),forward_strength_learnable=True)
            #self.proj_lif.forward_strength.data = torch.full((num_compartment,),1.0)    

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H*W
        #x = x.flatten(3)
        #T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,-1,H,W).contiguous()

        self.q_lif.soma_shape[0] = q_conv_out.shape[2]/self.num_compartment
        self.q_lif.soma_shape[1] = q_conv_out.shape[3]
        self.q_lif.soma_shape[2] = q_conv_out.shape[4]
        #if self.bn_alter == True:
            #self.q_lif.dend.compartment.c_sub = q_conv_out.shape[2]
        q_conv_out,_ = self.q_lif(q_conv_out)
        q = q_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,-1,H,W).contiguous()

        self.k_lif.soma_shape[0] = k_conv_out.shape[2]/self.num_compartment
        self.k_lif.soma_shape[1] = k_conv_out.shape[3]
        self.k_lif.soma_shape[2] = k_conv_out.shape[4]
        #if self.bn_alter == True:
            #self.k_lif.dend.compartment.c_sub = k_conv_out.shape[2]
        k_conv_out,_ = self.k_lif(k_conv_out)
        k = k_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,-1,H,W).contiguous()

        self.v_lif.soma_shape[0] = v_conv_out.shape[2]/self.num_compartment
        self.v_lif.soma_shape[1] = v_conv_out.shape[3]
        self.v_lif.soma_shape[2] = v_conv_out.shape[4]
        #if self.bn_alter == True:
            #self.v_lif.dend.compartment.c_sub = v_conv_out.shape[2]
        v_conv_out,_ = self.v_lif(v_conv_out)
        v = v_conv_out.flatten(3).transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous() ##
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T,B,-1,H,W).contiguous()
        #if self.bn_alter == True:
            #self.proj_lif.dend.compartment.c_sub = x.shape[2]
        self.proj_lif.soma_shape[0] = x.shape[2]/self.num_compartment
        self.proj_lif.soma_shape[1] = x.shape[3]
        self.proj_lif.soma_shape[2] = x.shape[4]
        x,_ = self.proj_lif(x)#.reshape(T,B,-1,H,W)

        return x

class TokenSpikingTransformer_dend(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1,dend=False,integer=False,num_compartment=2,multi=False,bn_alter=False,soma_astro=False):
        super().__init__()
        self.tssa = Token_dend_QK_Attention(dim, num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,
                                               proj_drop=drop_path,sr_ratio=sr_ratio, dend=dend,integer=integer,num_compartment=num_compartment,multi=multi,bn_alter=bn_alter,soma_astro=soma_astro)
        mlp_in_features = int(dim/num_compartment)
        self.mlp = MLP(in_features= mlp_in_features, hidden_features=dim, drop=drop,dend=dend,integer=integer,num_compartment=num_compartment,multi=multi,bn_alter=bn_alter,soma_astro=soma_astro)

    def forward(self, x):

        x = x + self.tssa(x)
        # print(torch.unique(x))
        x = x + self.mlp(x)
        # print(torch.unique(x))

        return x

class SpikingTransformer_dend(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1,dend=False,integer=False,num_compartment=2,multi=False,bn_alter=False,soma_astro=False):
        super().__init__()
        self.ssa = Spiking_dend_Self_Attention(dim, num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,
                                               proj_drop=drop_path,sr_ratio=sr_ratio,dend=dend,integer=integer,num_compartment=num_compartment,multi=multi,bn_alter=bn_alter,soma_astro=soma_astro)
        mlp_in_features = int(dim/num_compartment)
        self.mlp = MLP(in_features= mlp_in_features, hidden_features=dim, drop=drop,dend=dend,integer=integer,num_compartment=num_compartment,multi=multi,bn_alter=bn_alter,soma_astro=soma_astro)

    def forward(self, x):

        x = x + self.ssa(x)
        # print(torch.unique(x))
        x = x + self.mlp(x)
        # print(torch.unique(x))

        return x

class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,sps_integer=False):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if sps_integer==False:
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj_lif = soma.IntergerSoma()

        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        if sps_integer==False:
            self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj1_lif = soma.IntergerSoma()

        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        if sps_integer==False:
            self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj2_lif = soma.IntergerSoma()

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        if sps_integer==False:
            self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj3_lif = soma.IntergerSoma()

        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        if sps_integer==False:
            self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj_res_lif = soma.IntergerSoma()


    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        # x_feat = x.flatten(0, 1)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//8, W//8).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        x = x + x_feat # shortcut

        return x

class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,sps_integer=False):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        if sps_integer==False:
            self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj3_lif = soma.IntergerSoma()

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        if sps_integer==False:
            self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj4_lif = soma.IntergerSoma()

        self.proj_res_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        if sps_integer==False:
            self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:
            self.proj_res_lif = soma.IntergerSoma()

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None, pretrained_cfg_overlay=False, no_weight_decay = None, dend=False,integer=False,num_compartment=2,multi=False,sps_integer=False,concat=False,bn_alter=False,TET=False,soma_astro=False
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.TET = TET
        num_heads = [16, 16, 16]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        #
        patch_embed1 = PatchEmbedInit(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims // 2,sps_integer=sps_integer)

        stage1 = nn.ModuleList([TokenSpikingTransformer_dend(
            dim=embed_dims // 2, num_heads=num_heads[0], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, dend=dend,integer=integer,num_compartment=num_compartment,multi=multi,bn_alter=bn_alter,soma_astro=soma_astro)
            for j in range(1)])


        patch_embed2 = PatchEmbeddingStage(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims,sps_integer=sps_integer)


        stage2 = nn.ModuleList([SpikingTransformer_dend(
            dim=embed_dims, num_heads=num_heads[1], mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios, dend=dend,integer=integer,num_compartment=num_compartment,multi=multi,bn_alter=bn_alter,soma_astro=soma_astro)
            for j in range(1)])


        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"stage1", stage1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage2", stage2)


        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pose_embed'}

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        return None

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x,hook=None):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        if not self.TET:
            x = self.head(x.mean(0))
        else:
            x = self.head(x)    
        return x,hook


@register_model
def QKFormer_dend_dvs(pretrained=False, **kwargs):
    model = vit_snn(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=1,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,dend=True,multi=True,soma_astro=True,
        **kwargs #
    )
    model.default_cfg = _cfg()
    return model


from timm.models import create_model

if __name__ == '__main__':
    x = torch.randn(1, 1, 2, 128, 128).cuda()
    model = create_model(
        'QKFormer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    ).cuda()
    model.eval()

    from torchinfo import summary
    summary(model, input_size=(1, 1, 2, 128, 128))
    # y = model(x)
    # print(y.shape)
    # print('Test Good!')
