import torch 
from torch import nn

class position_wise_feed_forward(nn.Module):
    """
    位置前馈网络
    在Transformer中对序列中每个位置独立应用的前馈神经网络
    """
    def __init__(self, d_model, dropout = 0.1):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            dropout: dropout概率
        """
        super(position_wise_feed_forward, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 第一个线性层将维度扩展4倍
        self.fc1 = nn.Linear(d_model, d_model*4)
        # 第二个线性层将维度还原
        self.fc2 = nn.Linear(d_model*4, d_model)
        # ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入张量
            
        Returns:
            经过前馈网络处理后的张量
        """
        x = self.fc1(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc2(x)
        return x