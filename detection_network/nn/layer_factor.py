import torch
import torch.nn as nn


"""
- in_planes
- out_planes
- stride
"""
def conv3_3(in_planes,out_planes,stride=1,bias=False,dialation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dialation,
        dilation=dialation,
        bias=bias
    )

def conv1_1(in_planes,out_planes,stride=1,bias=False):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias
    )

def conv_bn(C_in,C_out,kernel_size,stride,padding,affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in,C_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
        nn.BatchNorm2d(C_out,affine=affine)
    )

def conv_bn_relu(C_in,C_out,kernel_size,stride,affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in,C_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
        nn.BatchNorm2d(C_out,affine=affine),
        nn.ReLU(inplace=False)
    )
def conv_bn_relu6(C_in,C_out,stride,affine=True):
    return nn.Sequential(
        nn.Conv2d(C_in,C_out,kernel_size=3,stride=stride,padding=padding,bias=False),
        nn.BatchNorm2d(C_out,affine=affine),
        nn.ReLU6(inplace=False)
    )



def conv_1_1_bn_relu6(in_planes,out_planes,stride):
    return nn.Sequential(
        nn.Conv2d(C_in,C_out,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(C_out),
        nn.ReLU6(inplace=True)
    )

class GAPConv_1_1(nn.Module):
    def __init__(self,C_in,C_out):
        super(GAPConv_1_1,self).__init__()
        self.conv1_1 = conv_bn_relu(C_in,C_out,1,stride=1,padding=0)

class InvertedResidual(nn.Module):
    def __init__(self,in_planes,out_planes,stride,expand_ratio):
        super(InvertedResidual,self).__init__()

        self.stride = stride
        assert stride in [1,2]

        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        

    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

"""

"""
class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x
"""

"""
class Skip(nn.Module):
    def __init__(self,C_in,C_out,stride):
        super(Skip,self).__init__()
        assert (C_out % C_in) == 0, "C_out must be divisible by C_in"
        self.repeats = (1,C_out//C_in,1,1)
    def forward(self,x):
        return x.repeat(self.repeats)

class Zero(nn.Module):
    def __init__(self,C_in,C_out,stride):
        super(Zero,self).__init__()
        assert (C_out % C_in) == 0, "C_out must be divisible by C_in"

class Adapt(nn.Module):
    def __init__(self,C_in0,C_in1,C_out,larger):


class ConcatReduce(nn.Module):
