from ppo import PPO
from reinforcement import REINFORCE

"""
returns:
    controller net that provides the sample() method
    gradient estimator
"""

def create_agent(
    enc_num_layers,
    num_ops,
    num_agg_ops,
    lstm_hidden_size,
    lstm_num_layers,
    dec_num_cells,
    cell_num_layers,
    cell_max_repeat,
    cell_max_stride,
    ctrl_lr,
    ctrl_baseline_decay,
    ctrl_agent,
    ctrl_version="cvpr"
):
    if ctrl_version == "cvpr":
        from rl.micro_controllers import 