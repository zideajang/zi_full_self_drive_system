import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import calc_prob, compute_critic_logits, sample_logits, torch_long


class MicroController(nn.Module):
    """
    With modification that indices and ops chosen with linear classifier
    Samples decoder structure and connections
    """

    def __init__(
        self,
        enc_num_layers,
        num_ops,
        lstm_hidden_size=100,
        lstm_num_layers=2,
        dec_num_cells=3,
        cell_num_layers=4,
        **kwargs
        ):
        super(MicroController,self).__init__()

        # 每个块(block)包含两个单元(cell)，分别对应 2 个输入
        self._num_cells_per_decoder_block = 2
        # 每单元(cell)接受 1 输入
        self._num_inputs_per_cell = 1
        # 对每个输入进行一个操作
        self._num_pos_per_cell_input = 1
        # 生成 1 个输出
        self._num_outputs_per_cell = 1

        # 在单元里，每层(除了第一层以外)接受 2
        self._num_inputs_per_cell_layer = 2
        self._num_ops_per_cell_layer_input =1

        # 每个单元层输出 1 个输出
        self._num_outputs_per_cell_layer = 1


        # 自定义配置
        self.cell_num_layers = cell_num_layers
        self.dec_num_cells = dec_num_cells
        self.enc_num_layers = enc_num_layers
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # 解码器连接的数量
        total_num_cell_inputs = (
            self._num_cells_per_decoder_block
            * self._num_inputs_per_cell
            * self.dec_num_cells
        )

        # 在单元中连接的数量
        total_num_cell_connections = self._num_inputs_per_cell

        self._action_len = (
            total_num_cell_inputs + total_num_cell_connections
        )

        # 控制器
        self.rnn = nn.LSTM(lstm_hidden_size, lstm_hidden_size,lstm_num_layers)
        self.enc_op = nn.Embedding(num_ops,lstm_hidden_size)
        self.linear_op = nn.Linear(lstm_hidden_size,num_ops)
        self.g_emb = nn.Parameter(torch.zeros)


        # 连接预测
        conn_fcs = []
        for i in range(self.dec_num_cells):


    @staticmethod
    def get_mock():
        arc_seq = [
            [[0],[1,2,3,4],[1,2,3,4],[1,2,3,4]],
            [[0,1],[2,3],[4,5]]
        ]
        entropy = 6
        log_prob = -1.4
        return arc_seq, entropy, log_prob

    @staticmethod
    def config2action(config):
        ctx, conns = config
        action = []

        for idx, cell in enumerate(ctx):
            if idx == 0:
                index = 0
                op = cell
                action += [index, op]
            else:
                index1, index2, op1,op2 = cell
                action += [index1,index2,op1,op2]

    """
    将动作转换为配置
    """
    @staticmethod
    def action2config(action,enc_end=0,dec_block=3,ctx_block=4):
        ctx = []
        for i in range(ctx_block):
            if i == 0:
                # 对于第一层只有输入和操作
                ctx.append([action[i],action[i+1]])
            else:
                # 对于随后层有 2 输入和 2 操作
                ctx.append([
                    action[(i-1) * 4 + 2],
                    action[(i-1) * 4 + 3],
                    action[(i-1) * 4 + 4],
                    action[(i-1) * 4 + 5],
                ])

        conns = []
        for i in range(dec_block):
            conns.append(
                [
                    action[4 * (ctx_block - 1) + 2 + i * 2],
                    action[4 * (ctx_block - 1) + 2 + i * 2 + 1],
                ]
            )
        return [ctx,conns]

class TemplateController(nn.Module):
    def __init__(self):
        pass

    def forward(self,config=None):
