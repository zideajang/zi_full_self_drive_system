# 先写测试网络结构
import logging

import torch
import torch.nn as nn
from torch.nn.encoder import create_encoder


class DetectionSegmenter(nn.Module):
    def __init__(self,encoder,decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

def setup_loggr():
    logger = logging.getLogger(__name__)

defaults_config = {
    "random_seed":106,
    "arch_writer":"output/genotypes.out",
    "num_tasks":10,
}


def main():
    """
    先做训练的控制器
    """
    setup_loggr()
    # logger.info("hello ZiFSD")


    # 定义输出结构的文件
    arch_writer = open(defaults_config["random_seed"],"w")
    # class_num = 1000

    segmentation_criterion = nn.NLLLoss2d(ignore_index=255).cuda()




    # 搜索空间
    sample = ((decoder_config),reward,entropy,log_prob)


    # 定义智能代理
    


if __name__ == "__main__":
    main()