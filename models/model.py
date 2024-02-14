import torch
import torch.nn as nn

class PreTrainingModel(nn.Module):
    def __init__(self, convnet, head):
        super(PreTrainingModel, self).__init__()
        self.convnet = convnet
        self.head = head
        
    def forward(self, inputs: dict):
        x = self.convnet(inputs)
#         print(x.shape)
        outputs = self.head(x)
        return outputs