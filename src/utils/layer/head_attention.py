import torch
from torch import nn
import math

class mutihead_attention(nn.Module):
    """
    多头注意力机制实现
    该模块实现Transformer中的多头自注意力机制
    """
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力模块
        
        Args:
            d_model: 模型的维度大小
            num_heads: 注意力头的数量
        """
        super(mutihead_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 定义查询、键、值的线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # softmax用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        """
        前向传播过程
        
        Args:
            query: 查询向量
            key: 键向量
            value: 值向量
            
        Returns:
            经过多头注意力计算后的输出向量
        """
        batch_size = query.shape[0]

        # 对输入进行线性变换
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        # 计算每个注意力头的维度
        self.head_dim = self.d_model//self.num_heads
        # 将输入重新组织为多头形式: (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, self.num_heads,-1, self.head_dim).transpose(1,2)
        k = k.view(batch_size, self.num_heads,-1, self.head_dim).transpose(1,2)
        v = v.view(batch_size, self.num_heads,-1, self.head_dim).transpose(1,2)
        
        # 计算注意力分数并进行缩放
        attention_scores =q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        # 应用softmax获取注意力权重
        attention_weights = self.softmax(attention_scores)
        
        # 使用注意力权重对值向量进行加权求和
        attention_value = torch.matmul(attention_weights,v)
        
        # 将多头结果重新组合为原始维度
        attention_value = attention_value.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        return attention_value