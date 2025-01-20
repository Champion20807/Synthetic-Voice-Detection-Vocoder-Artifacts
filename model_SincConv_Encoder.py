import torch
import torch.nn as nn
import torch.nn.functional as F
from sincnet import SincConv1d  # 假設你有安裝 sincnet 庫

class RawNetWithSincConvAndTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(RawNetWithSincConvAndTransformer, self).__init__()
        
        self.device = device


        self.sinc_conv = SincConv1d(
            in_channels=1, 
            out_channels=d_args['num_filters'], 
            kernel_size=d_args['kernel_size'], 
            sample_rate=d_args['sample_rate']
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        

        # Transformer encoder
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
        
        # Fully connected layers
        self.fc1 = nn.Linear(d_args['hidden_size'], d_args['nb_fc_node'])
        self.fc2_binary = nn.Linear(d_args['nb_fc_node'], 2, bias=True)
        self.fc2_multi = nn.Linear(d_args['nb_fc_node'], 7, bias=True)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input: (batch, seq_len)
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)  # (batch, 1, seq_len)
        
        # SincConv + Max Pooling
        x = self.sinc_conv(x)  # (batch, num_filters, new_seq_len)
        x = self.pool(x)      
        
        # Transformer encoder + Mean Pooling
        x = x.permute(2, 0, 1)  # (batch, num_filters, new_seq_len) -> (new_seq_len, batch, num_filters)
        x = self.transformer_encoder(x)  # (new_seq_len, batch, hidden_size)
        x = x.mean(dim=0)  # (batch, hidden_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x_binary = self.fc2_binary(x)
        x_multi = self.fc2_multi(x)
        
        output_binary = self.logsoftmax(x_binary)
        output_multi = self.logsoftmax(x_multi)
        
        return output_binary, output_multi
