import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class RawNetWithTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(RawNetWithTransformer, self).__init__()
        
        self.device = device
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_args['hidden_size'], 
            nhead=d_args['num_heads'], 
            dim_feedforward=d_args['ffn_dim'], 
            dropout=d_args['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=d_args['num_layers']
        )
        
        self.fc1 = nn.Linear(d_args['hidden_size'], d_args['nb_fc_node'])
        self.fc2_binary = nn.Linear(d_args['nb_fc_node'], 2, bias=True)
        self.fc2_multi = nn.Linear(d_args['nb_fc_node'], 7, bias=True)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input: (batch, seq_len)
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, len_seq, 1)  # Add feature dimension
        
        # Transform input for Transformer Encoder (batch, seq_len, feature_dim)
        x = x.permute(1, 0, 2)  # (batch, seq_len, feature_dim) -> (seq_len, batch, feature_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch, hidden_size)
        x = x.mean(dim=0)  # Average pooling over sequence length (batch, hidden_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x_binary = self.fc2_binary(x)
        x_multi = self.fc2_multi(x)
        
        output_binary = self.logsoftmax(x_binary)
        output_multi = self.logsoftmax(x_multi)
        
        return output_binary, output_multi