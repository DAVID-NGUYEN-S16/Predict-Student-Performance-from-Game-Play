import torch
import torch.nn as nn
from .time_imbedding import TimeEmbedding
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, input_dims, d_model, n_blocks=4):
        super(ConvNet, self).__init__()
        self.input_dims = input_dims
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.event_embedding = nn.Embedding(input_dims['event_name_name'], d_model, padding_idx=0)
        self.room_embedding = nn.Embedding(input_dims['room_fqid'], d_model, padding_idx=0)
        self.text_embedding = nn.Embedding(input_dims['text'], d_model, padding_idx=0)
        self.fqid_embedding = nn.Embedding(input_dims['fqid'], d_model, padding_idx=0)
        self.duration_embedding = TimeEmbedding(n_blocks=n_blocks, d_model=d_model, dropout_rate=0.2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=24,
            nhead=8,
            dim_feedforward=24,
            dropout=0.1,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, inputs):
        inputs['event_name_name'] = inputs['event_name_name'].long()
        inputs['room_fqid'] = inputs['room_fqid'].long()
        inputs['text'] = inputs['text'].long()
        inputs['fqid'] = inputs['fqid'].long()
        inputs['duration'] = inputs['duration'].to(torch.float32)
        event = self.event_embedding(inputs['event_name_name'])
        room = self.room_embedding(inputs['room_fqid'])
        text = self.text_embedding(inputs['text'])
        fqid = self.fqid_embedding(inputs['fqid'])
        
        inputs['duration'] = inputs['duration'].unsqueeze(0).unsqueeze(0)
        duration = self.duration_embedding(inputs['duration']).squeeze(-1)
        sum_cag = (event + room + text + fqid)

        x = sum_cag*duration.unsqueeze(1).expand(sum_cag.shape)
        x = self.encoder(x)
        outputs = self.gap(x)
        return outputs.transpose(2, 0)
    
class SimpleHead(nn.Module):
    def __init__(self, n_units, n_outputs, dropout_rate=0.2):
        super(SimpleHead, self).__init__()
        self.ffs = nn.ModuleList([nn.Linear(n_units[i-1], n_units[i]) for i in range(1, len(n_units))])
        self.out = nn.Linear(n_units[-1], n_outputs)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        x = inputs.transpose(2, 1)
        # print(x.shape)
        for ff in self.ffs:
            x = F.leaky_relu(ff(x))
            x = self.dropout(x)
        outputs = torch.sigmoid(self.out(x))
        return outputs
    