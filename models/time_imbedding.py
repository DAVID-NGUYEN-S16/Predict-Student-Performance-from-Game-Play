import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super(ConvBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, padding=2)
        self.gelu = nn.GELU("tanh")
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, inputs):
        x = self.conv1d(inputs)
        x = self.gelu(x)
        x = x + inputs
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        
        outputs = self.dropout(x)
        return outputs

class TimeEmbedding(nn.Module):
    def __init__(self, n_blocks, d_model, dropout_rate):
        super(TimeEmbedding, self).__init__()
        self.conv_blocks = nn.ModuleList([ConvBlock(d_model, dropout_rate=dropout_rate) for _ in range(n_blocks)])
        self.d_model = d_model
    
    def forward(self, inputs):
        x = inputs.view(-1, 1).unsqueeze(-1)
        b, r, c = x.shape
        x = x.expand(b, self.d_model , c)
        for conv_block in self.conv_blocks:

            x = conv_block(x)

        return x