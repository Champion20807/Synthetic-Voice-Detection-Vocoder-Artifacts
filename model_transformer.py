import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.utils import data
from transformers import Wav2Vec2Model

class RawNetWithTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(RawNetWithTransformer, self).__init__()
        
        self.device = device

        self.Sinc_conv = SincConv(device=self.device,
                                  out_channels=d_args['filts'][0],
                                  kernel_size=d_args['first_conv'],
                                  in_channels=d_args['in_channels'])
        
        self.first_bn = nn.BatchNorm1d(num_features=d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        
        # 保留原有的 Residual Blocks
        self.block0 = Residual_block(nb_filts=d_args['filts'][1], first=True)
        self.block1 = Residual_block(nb_filts=d_args['filts'][1])
        self.block2 = Residual_block(nb_filts=d_args['filts'][2])
        self.block3 = Residual_block(nb_filts=d_args['filts'][2])
        self.block4 = Residual_block(nb_filts=d_args['filts'][2])
        self.block5 = Residual_block(nb_filts=d_args['filts'][2])
        
        self.bn_before_transformer = nn.BatchNorm1d(num_features=d_args['filts'][2][-1])
        
        # 替換為 Wav2Vec2 預訓練模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        self.fc1 = nn.Linear(self.wav2vec2.config.hidden_size, d_args['nb_fc_node'])
        self.fc2_binary = nn.Linear(d_args['nb_fc_node'], 2, bias=True)
        self.fc2_multi = nn.Linear(d_args['nb_fc_node'], 7, bias=True)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        
        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.bn_before_transformer(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) → (batch, time, filt)
        
        # 使用 Wav2Vec2 進行特徵提取
        x = self.wav2vec2(x).last_hidden_state
        x = x.mean(dim=1)  # 平均池化

        x = self.fc1(x)
        
        x_binary = self.fc2_binary(x)
        x_multi = self.fc2_multi(x)
        
        output_binary = self.logsoftmax(x_binary)
        output_multi = self.logsoftmax(x_multi)
        
        return output_binary, output_multi
