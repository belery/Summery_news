import torch
import torch.nn as nn
import math
from utils.helpers import get_pad_mask
class TransformerSummarizer(nn.Module):
    """
    基于Transformer的新闻摘要模型
    """
    
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout):
        """
        初始化模型
        
        Args:
            vocab_size (int): 词汇表大小
            d_model (int): 模型维度
            nhead (int): 多头注意力头数
            num_encoder_layers (int): 编码器层数
            num_decoder_layers (int): 解码器层数
            dim_feedforward (int): 前馈网络维度
            max_seq_length (int): 最大序列长度
        """
        super(TransformerSummarizer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = self.create_pos_encoding(max_seq_length, d_model)
        
        # Transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=False  # PyTorch Transformer默认使用seq_first格式
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def create_pos_encoding(self, max_len, d_model):
        """
        创建位置编码
        
        Args:
            max_len (int): 最大序列长度
            d_model (int): 模型维度
            
        Returns:
            torch.Tensor: 位置编码张量
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        前向传播
        
        Args:
            src (torch.Tensor): 源序列
            tgt (torch.Tensor): 目标序列
            src_mask (torch.Tensor): 源序列掩码
            tgt_mask (torch.Tensor): 目标序列掩码
            src_key_padding_mask (torch.Tensor): 源序列填充掩码
            tgt_key_padding_mask (torch.Tensor): 目标序列填充掩码
            memory_key_padding_mask (torch.Tensor): 记忆填充掩码
            
        Returns:
            torch.Tensor: 模型输出
        """
        # Embedding and position encoding
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        
        # Add position encoding
        src_embed = src_embed + self.pos_encoding[:, :src_embed.size(1), :]
        tgt_embed = tgt_embed + self.pos_encoding[:, :tgt_embed.size(1), :]
        
        # Transpose to match PyTorch Transformer requirement: (seq_len, batch_size, d_model)
        src_embed = src_embed.transpose(0, 1)
        tgt_embed = tgt_embed.transpose(0, 1)
        
        device = src_embed.device
        
        # Move masks to device (no transpose needed)
        # According to PyTorch documentation, when batch_first=False (default),
        # src_key_padding_mask should be (batch_size, seq_len)
        # However, internally PyTorch will handle the dimension properly
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(device)
        elif src_key_padding_mask is not None:
            # If memory_key_padding_mask is not provided, use src_key_padding_mask
            memory_key_padding_mask = src_key_padding_mask

        # Forward through transformer
        output = self.transformer(src_embed, tgt_embed, 
                                  src_mask=src_mask, 
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask, 
                                  tgt_key_padding_mask=tgt_key_padding_mask, 
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Transpose back to (batch_size, seq_len, d_model)
        output = output.transpose(0, 1)
        output = self.output_layer(output)
        return output
    
    def generate(self, src, max_length, start_token_id, end_token_id):
        """
        生成摘要
        
        Args:
            src (torch.Tensor): 源序列
            max_length (int): 最大生成长度
            start_token_id (int): 起始标记ID
            end_token_id (int): 结束标记ID
            
        Returns:
            torch.Tensor: 生成的摘要序列
        """
        pass