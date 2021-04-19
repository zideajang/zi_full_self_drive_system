from collections import namedtuple

"""

"""

Genotype = namedtuple("Genotype","encoder decoder")


OP_NAMES = [
    "conv1_1",
    "conv3_3",
    "sep_conv_3_3",
    "sep_conv_5_5",
    "global_average_pool",
    "conv3_3_dil3",
    "conv3_3_dil6",
    "sep_conv_3_3_dil3",
    "sep_conv_5_5_dil6",
    "skip_connect",
    "none"
]

OP_NAMES_WACV = [
    "sep_conv_3_3",
    "sep_conv_5_5",
    "global_average_pool",
    "max_pool_3_3",
    "sep_conv_5_5_dil6",
    "skip_connect"
]


AGG_OP_NAMES = [
    "psum",
    "cat"
]