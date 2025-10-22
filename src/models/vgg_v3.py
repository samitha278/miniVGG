import torch
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'




@dataclass
class Config():
    in_ch : int = 3
    n_embd : int = 14700
    n_class : int = 100
    dropout : int = 0.1
    


class minVGG(nn.Module):

    def __init__(self,config):
        super().__init__()



        self.conv_1 = nn.Conv2d(config.in_ch, 2*config.in_ch,5)
        self.bn1 = nn.BatchNorm2d(2*config.in_ch)     # Batch Norm 1
        self.pooling_1 = nn.MaxPool2d(3)

        self.conv_2 = nn.Conv2d(2* config.in_ch, 4*config.in_ch,3)
        self.bn2 = nn.BatchNorm2d(4*config.in_ch)     # Batch Norm 2
        self.pooling_2 = nn.MaxPool2d(2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.mlp = MLP(config)


    def forward(self,x,targets = None):

        B,C,H,W = x.shape

        #1
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pooling_1(x)

        #2
        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pooling_2(x)

        x = self.flatten(x)
        
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            return logits , F.cross_entropy(logits,targets)
        
        return logits


class MLP(nn.Module):


    def __init__(self,config):
        super().__init__()
        
        # pre layer norm
        self.ln = nn.LayerNorm(config.n_embd)
        self.layer = nn.Linear(config.n_embd,4*config.n_embd)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4*config.n_embd,config.n_class)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self,x):
        
        x = self.ln(x)
        x = self.layer(x)
        x = self.relu(x)
        x = self.proj(x)
        x = self.dropout(x)

        return x
  