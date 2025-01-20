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

        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        self.fc1 = nn.Linear(self.wav2vec2.config.hidden_size, d_args['nb_fc_node'])
        self.fc2_binary = nn.Linear(d_args['nb_fc_node'], 2, bias=True)
        self.fc2_multi = nn.Linear(d_args['nb_fc_node'], 7, bias=True)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input -> (batch, seq_len)

        # Wav2Vec2 + Mean Pooling
        x = self.wav2vec2(x).last_hidden_state  # (batch, seq_len, hidden_size)
        x = x.mean(dim=1)  # (batch, hidden_size)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch, nb_fc_node)
        x_binary = self.fc2_binary(x)  # (batch, 2)
        x_multi = self.fc2_multi(x)  # (batch, 7)
        
        output_binary = self.logsoftmax(x_binary)
        output_multi = self.logsoftmax(x_multi)
        
        return output_binary, output_multi
