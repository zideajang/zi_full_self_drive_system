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

def conv_bn(channel_in,channel_out,kernel_size,stride,padding,affine=True):
    return nn.Sequential(
        nn.Conv2d(channel_in,channel_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
        nn.BatchNorm2d(channel_out,affine=affine)
    )

def conv_bn_relu(channel_in,channel_out,kernel_size,stride,affine=True):
    return nn.Sequential(
        nn.Conv2d(channel_in,channel_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
        nn.BatchNorm2d(channel_out,affine=affine),
        nn.ReLU(inplace=False)
    )
def conv_bn_relu6(channel_in,channel_out,stride,affine=True):
    return nn.Sequential(
        nn.Conv2d(channel_in,channel_out,kernel_size=3,stride=stride,padding=padding,bias=False),
        nn.BatchNorm2d(channel_out,affine=affine),
        nn.ReLU6(inplace=False)
    )



def conv_1_1_bn_relu6(in_planes,out_planes,stride):
    return nn.Sequential(
        nn.Conv2d(channel_in,channel_out,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(channel_out),
        nn.ReLU6(inplace=True)
    )

class GAPConv_1_1(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(GAPConv_1_1,self).__init__()
        self.conv1_1 = conv_bn_relu(channel_in,channel_out,1,stride=1,padding=0)

"""
池化层
"""
class Pool(nn.Module):
    def __init__(self,channel_in,channel_out,stride,repeats,kernel_size,mode):
        super(Pool,self).__init__()
        self.conv1_1 = conv_bn(channel_in,channel_out,1,1,0)
        if mode == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size=kernel_size,
                padding=(kernel_size//2),
                count_include_pad=False
            )
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size,stride=stride,padding=kernel_size//2)
        else:
            raise ValueError(f"Unknown pooling method{mode}")

class DilConv(nn.Module):

    def __init__(self,channel_in,channel_out,kernel_size,stride,padding,dilation,affine=True):
        super(DilConv,self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channel_in,
                channel_out,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=channel_in,
                bias=False,
            ),
            nn.Conv2d(channel_in,channel_out,kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(channel_out,affine=affine),
        )

    def forward(self,x):
        return self.op(x)

class SepConv(nn.Module):

    def __init__(
        self,
        channel_in,
        channel_out,
        kernel_size,
        stride,
        padding,
        dilation=1,
        affine=True,
        repeats=1
        )

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
    def __init__(self,channel_in,channel_out,stride):
        super(Skip,self).__init__()
        assert (channel_out % channel_in) == 0, "channel_out must be divisible by channel_in"
        self.repeats = (1,channel_out//channel_in,1,1)
    def forward(self,x):
        return x.repeat(self.repeats)

class Zero(nn.Module):
    def __init__(self,channel_in,channel_out,stride):
        super(Zero,self).__init__()
        assert (channel_out % channel_in) == 0, "channel_out must be divisible by channel_in"

class Adapt(nn.Module):
    def __init__(self,channel_in0,channel_in1,channel_out,larger):


class ConcatReduce(nn.Module):

