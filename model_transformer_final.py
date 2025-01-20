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

        # 使用 Wav2Vec2 作為音訊特徵提取器
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # 定義全連接層
        self.fc1 = nn.Linear(self.wav2vec2.config.hidden_size, d_args['nb_fc_node'])
        self.fc2_binary = nn.Linear(d_args['nb_fc_node'], 2, bias=True)
        self.fc2_multi = nn.Linear(d_args['nb_fc_node'], 7, bias=True)
        
        # 使用 LogSoftmax 作為輸出
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        # 假設輸入是 (batch, time) 格式的音訊訊號
        x = self.wav2vec2(x).last_hidden_state  # 提取 Wav2Vec2 的隱藏層輸出
        x = x.mean(dim=1)  # 平均池化
        
        # 通過全連接層
        x = self.fc1(x)
        x_binary = self.fc2_binary(x)
        x_multi = self.fc2_multi(x)
        
        # 分別輸出二元分類和多元分類
        output_binary = self.logsoftmax(x_binary)
        output_multi = self.logsoftmax(x_multi)
        
        return output_binary, output_multi
