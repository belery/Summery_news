import torch
from torch import nn
import math

# 修正导入语句，使其与实际文件名匹配
from head_attention import mutihead_attention
from position_wise_feed_forwar import position_wise_feed_forward
    
class encoder_layer(nn.Module):
    """
    Transformer编码器层
    包含多头注意力和位置前馈网络两个子层
    """
    def __init__(self, d_model, num_heads, dropout):
        """
        初始化编码器层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super(encoder_layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 位置前馈网络
        self.ffn_hidden = position_wise_feed_forward(d_model, dropout)
        # 多头注意力机制
        self.attention = mutihead_attention(d_model, num_heads)
        # dropout层用于正则化
        self.dropout = nn.Dropout(dropout)
        # LayerNorm用于标准化
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入序列张量
            
        Returns:
            经过编码器层处理后的张量
        """
        # 保存残差连接的原始输入
        temp_x = x
        # 多头自注意力子层
        x = self.attention.forward(x,x,x)
        x = self.dropout(x)
        # 残差连接和LayerNorm
        x = self.norm1(temp_x + x)
        
        # 保存残差连接的输入
        temp_x = x
        # 位置前馈网络子层
        x = self.ffn_hidden.forward(x)
        x = self.dropout(x)
        # 残差连接和LayerNorm
        x = self.norm2(temp_x + x)
        return x




if __name__ == '__main__':
    # 创建测试数据 (batch_size=10, seq_length=10, d_model=10)
    test = torch.randn(10,10,10)
    # 测试编码器层
    test = encoder_layer(10, 2, 0.1).forward(test)
    print(test.shape)