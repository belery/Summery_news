import torch
from torch import nn
import math
import torch.nn.functional as F

from torch import Tensor

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding,self).__init__(vocab_size, d_model, padding_idx=1)


class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(positional_encoding,self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device = device)
        pos=pos.float().unsqueeze(dim = 1)
        _2i = torch.arange(0, d_model, step = 2, device = device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:,:seq_len, :]
        
class transformer_embeding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device, dropout):
        super(transformer_embeding,self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = positional_encoding(d_model, max_len, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.token_embedding(x) + self.position_encoding(x)
        return self.dropout(out)