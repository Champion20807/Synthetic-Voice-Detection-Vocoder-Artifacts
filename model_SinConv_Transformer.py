import torch
import torch.nn as nn
import torch.nn.functional as F
from sincnet import SincConv1d  # 確保 SincConv1d 的實現可用

class RawNetWithSincConvTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(RawNetWithSincConvTransformer, self).__init__()
        
        self.device = device
        
        # SincConv 層
        self.sincconv = SincConv1d(
            in_channels=1, 
            out_channels=256,  # 將 feature_dim 提升至 256
            kernel_size=129,  # 自行調整適合的 kernel 大小
            sample_rate=16000  # 確保這裡的設定與你的音頻數據一致
        )
        self.sincconv_bn = nn.BatchNorm1d(256)  # 批量標準化

        # Transformer 層
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
        
        # Fully connected 層
        self.fc1 = nn.Linear(d_args['hidden_size'], d_args['nb_fc_node'])
        self.fc2_binary = nn.Linear(d_args['nb_fc_node'], 2, bias=True)
        self.fc2_multi = nn.Linear(d_args['nb_fc_node'], 7, bias=True)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input: (batch, seq_len)
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)  # Add channel dimension (batch, 1, seq_len)

        # SincConv + BN
        x = self.sincconv(x)  # (batch, 256, new_seq_len)
        x = self.sincconv_bn(x)
        x = F.relu(x)

        # Permute for Transformer (batch, seq_len, feature_dim)
        x = x.permute(2, 0, 1)  # (batch, 256, seq_len) -> (seq_len, batch, 256)
        x = self.transformer_encoder(x)  # (seq_len, batch, hidden_size)
        x = x.mean(dim=0)  # Average pooling over sequence length (batch, hidden_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x_binary = self.fc2_binary(x)
        x_multi = self.fc2_multi(x)
        
        output_binary = self.logsoftmax(x_binary)
        output_multi = self.logsoftmax(x_multi)
        
        return output_binary, output_multi
