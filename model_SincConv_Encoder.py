import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

class RawNetWithTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(RawNetWithTransformer, self).__init__()

        self.device = device

        # 初始化 SincConv 層
        self.Sinc_conv = SincConv(device=self.device,
                                  out_channels=d_args['filts'][0],
                                  kernel_size=d_args['first_conv'], 
                                  in_channels=d_args['in_channels'])

        # 預訓練的 Transformer 模型（例如 BERT）
        self.bert_config = BertConfig(
            hidden_size=d_args['gru_node'],  # 與原先 GRU 隱藏層的大小對應
            num_attention_heads=12,  # 設定為12頭注意力
            num_hidden_layers=6,  # 設定層數
            intermediate_size=d_args['nb_fc_node'],  # 對應於全連接層的大小
            max_position_embeddings=512,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1
        )
        
        self.bert_model = BertModel(self.bert_config)

        # 輸出層
        self.fc1_binary_gru = nn.Linear(d_args['gru_node'], d_args['nb_fc_node'])
        self.fc2_binary_gru = nn.Linear(d_args['nb_fc_node'], 2)
        self.fc1_multi_gru = nn.Linear(d_args['gru_node'], d_args['nb_fc_node'])
        self.fc2_multi_gru = nn.Linear(d_args['nb_fc_node'], 7)

        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        # SincConv 預處理
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        # 將輸入轉換為適合 Transformer 的格式
        # 在這裡，`x` 應該是 `(batch_size, seq_len, input_size)`
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len) -> (batch_size, seq_len, input_size)
        
        # Transformer 輸出
        transformer_output = self.bert_model(inputs_embeds=x)[0]  # 只取 transformer 的輸出部分
        
        # 取出最後一層的輸出作為序列的表示
        x = transformer_output[:, -1, :]  # 取序列的最後一個時間步的輸出

        # 進行二分類和多分類
        x_binary = self.fc1_binary_gru(x)
        x_binary = self.fc2_binary_gru(x_binary)

        output_binary = self.logsoftmax(x_binary)

        x_multi = self.fc1_multi_gru(x)
        x_multi = self.fc2_multi_gru(x_multi)

        output_multi = self.logsoftmax(x_multi)

        return output_binary, output_multi
