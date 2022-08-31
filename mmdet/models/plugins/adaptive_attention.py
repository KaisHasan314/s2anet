import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init


class AdaptiveAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 r=16,
                 s=16):
        super(AdaptiveAttention, self).__init__()
        self.c = in_channels
        self.global_max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        compressed_features1 = round(in_channels/s)
        self.fc1 = nn.Linear(in_channels, compressed_features1)
        self.fc2 = nn.Linear(compressed_features1, in_channels)
        self.sigmoid = nn.Sigmoid()

        compressed_features2 = round(2*in_channels/s)
        self.fc3 = nn.Linear(in_channels*2, compressed_features2)
        self.fc4 = nn.Linear(compressed_features2, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.layers = [
            self.fc1, self.fc2, self.fc3, self.fc4
        ]

    def forwardLeft(self, x):
        x = self.global_avg_pooling(x)
        x = x.permute((0, 2, 3, 1))
        x = F.relu(self.fc1(x))
        logits = self.sigmoid(self.fc2(x))
        logits = logits.permute((0, 3, 1, 2))
        return logits

    def forwardRight(self, x):
        x = self.global_max_pooling(x)
        x = x.permute((0, 2, 3, 1))
        x = F.relu(self.fc1(x))
        logits = self.sigmoid(self.fc2(x))
        logits = logits.permute((0, 3, 1, 2))
        return logits
    
    def forward_domain_attention_unit(self, x):
        x = x.permute((0, 2, 3, 1))
        x = F.relu(self.fc3(x))
        logits = self.softmax(self.fc4(x))
        logits = logits.permute((0, 3, 1, 2))
        return logits

    def forward(self, x):
        # print(f'x.shape:{x.shape}')
        
        x_l = self.forwardLeft(x)
        x_r = self.forwardRight(x)
        # print(f'x_l.shape:{x_l.shape}')
        # print(f'x_r.shape:{x_r.shape}')
        assert x_l.shape[1:] == (self.c, 1, 1)
        assert x_r.shape[1:] == (self.c, 1, 1)

        c_x_l = torch.cat([x_r, x_l], dim=1)
        c_x_r = torch.cat([x_r, x_l], dim=3)

        # print(f'c_x_l.shape:{c_x_l.shape}')
        #print(f'c_x_r.shape:{c_x_r.shape}')

        assert c_x_l.shape[1:] == (self.c*2, 1, 1)
        assert c_x_r.shape[1:] == (self.c, 1, 2)

        x_da = self.forward_domain_attention_unit(c_x_l)
        
        assert x_da.shape[1:] == (2, 1, 1)
        x_da = torch.reshape(x_da, (x_da.shape[0], 2, 1))
        c_x_r = torch.reshape(c_x_r, (c_x_r.shape[0], self.c, 2))
        
        x = torch.matmul(c_x_r, x_da).unsqueeze(-1)
        # print(f'x.shape:{x.shape}')
        assert x.shape[1:] == (self.c, 1, 1)

        return x

    def init_weights(self):
        for layer in self.layers:
            normal_init(layer, std=0.01)




