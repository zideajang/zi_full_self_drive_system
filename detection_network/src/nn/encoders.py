import torch
import torch.nn as nn

from layer_factor import InvertedResidual,conv_bn_relu6

model_path = {
    "mbv2_voc":"input/weights/"
}

# __all__ = ["mobile_network_v2"]



"""
定义 Mobile Network 
"""
class MobileNetworkV2(nn.Module):
    mobilenetwork_config = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    # 输入通道数
    in_planes = 32
    num_layers = len(mobilenetwork_config)

    def __init__(self,width_mult=1.0,return_layers=[1, 2, 4, 6]):
        super(MobileNetworkV2,self).__init__()
        self.return_layers = return_layers
        self.max_layer = max(self.return_layers)
        self.out_size = [
            self.mobilenetwork_config[layer_idx][1] for layer_idx in self.return_layers
        ]

        input_channel = int(self.in_planes * width_mult)

        self.layer_1 = conv_bn_relu6(3, input_channel, 2)
        for layer_idx, (t,c,n,s) in enumerate(self.mobilenetwork_config[:self.max_layer + 1]):
            output_channel = int(c * width_mult)
            features = []

            for i in range(n):
                if i == 0:
                    features.append(
                        InvertedResidual(input_channel,output_channel,s,t)
                    )
                else:
                    

    def forward(self,x):
        outs = []
        x = self.layer1(x)
        for layer_idx in range(self.max_layer + 1):
            x = getattr(self,f"layer{layer_idx+2}")(x)
            outs.append(x)
        return [outs[layer_idx] for layer_idx in self.return_layers]


def mobile_network_v2(pretrained=False,**kwargs):
    # 构造 MobileNet-v2 模型作为编码器
    model = MobileNetworkV2(**kwargs)
    if pretrained:
        model.load_state_dict(
            torch.load(model_path),strict=False
        )
    return model
def create_encoder(pretrained="voc",ctrl_version="cvpr",**kwargs):
    """创建编码器"""
    layers_res = [1,2,4,6] if ctrl_version == "cvpr" else [1,2]
    return mobile_network_v2(pretrained=pretrained,)