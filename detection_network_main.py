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


def main():
    """
    先做训练的控制器
    """
    setup_loggr()
    logger.info("hello ZiFSD")


if __name__ == "__main__":
    main()