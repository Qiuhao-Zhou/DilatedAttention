import torch
import torch.nn as nn

import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
import os, time
import functools

curr_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(curr_dir, "src")
_build_path = os.path.join(curr_dir, "build")
os.makedirs(_build_path, exist_ok=True)
pyda = load(name="pyda",
            extra_cflags=["-O3"],
            build_directory=_build_path,
            verbose=True,
            sources = [os.path.join(_src_path, f) for f in [
                "lib_da.cpp", "da.cu"
                ]],
            extra_cuda_cflags=["--expt-extended-lambda"])

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class DA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        size = (n, 9, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)

        pyda.da_forward_cuda(t, f, weight)
        
        # Output
        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)

        pyda.da_backward_cuda(dw.contiguous(), t, f, dt, df)

        _check_contiguous(dt, df)

        return dt, df

class DA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)
        pyda.da_map_forward_cuda(weight, g, out)
        
        # Output
        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)

        pyda.da_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)

        _check_contiguous(dw, dg)

        return dw, dg

da_weight = DA_Weight.apply
da_map = DA_Map.apply


class DilatedAttention(nn.Module):
    """ Dilated Attention Module"""
    def __init__(self,in_dim):
        super(DilatedAttention,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        #self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        #out = self.gamma*out + x
        return out



__all__ = ["DilatedAttention", "da_weight", "da_map"]


if __name__ == "__main__":
    ca = DilatedAttention(256).cuda()
    x = torch.zeros(1, 8, 10, 10).cuda() + 1
    y = torch.zeros(1, 8, 10, 10).cuda() + 2
    z = torch.zeros(1, 64, 10, 10).cuda() + 3
    out = ca(x, y, z)
    print (out)
