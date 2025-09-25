import argparse
from data.cnews_data_processor import CNewsDataProcessor
from model.transformer_summarizer import TransformerSummarizer
from model.trainer import ModelTrainer
from torch.utils.data import DataLoader, Dataset
import jieba
from utils.TxetSummary import TextSummaryDataset
import torch

def main():
    """
    主程序入口
    """
    train_data_path = "../../cnews_副本/train.src"
    test_data_path = "../../cnews_副本/test.src"
    val_data_path = "../../cnews_副本/valid.src"
    train_tgt_path = "../../cnews_副本/train.tgt"
    val_tgt_path = "../../cnews_副本/valid.tgt"
    epochs = 20  # 减少训练轮数
    batch_size = 16  # 减小批次大小
    max_len = 512  # 减小序列长度
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备: ", device)

    train_model(train_data_path, test_data_path, val_data_path, max_len, epochs, batch_size, learning_rate, train_tgt_path, val_tgt_path, device)

def train_model(train_data_path, test_data_path, val_data_path, max_len, epochs, batch_size, learning_rate, train_tgt_path, val_tgt_path, device):
    """
    训练模型
    
    Args:
        args: 命令行参数
    """
    # 初始化数据处理器
    print("初始化数据处理器...")
    data_processor = CNewsDataProcessor(train_data_path, val_data_path, train_tgt_path,val_tgt_path, max_len)
    
    # 加载和预处理数据
    print("加载数据...")
    train_data, val_data = data_processor.load_data()
    print(f"训练数据数量: {len(train_data)}, 验证数据数量: {len(val_data)}")

    print("创建训练数据集...")
    train_dataset = TextSummaryDataset(train_data, jieba, max_len)
    print("创建验证数据集...")
    val_dataset = TextSummaryDataset(val_data, jieba, max_len)

    # 获取词汇表大小
    vocab_size = len(train_dataset.word2idx)
    print(f"词汇表大小: {vocab_size}")

    #创建数据加载器
    print("创建数据加载器...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 初始化模型
    print("初始化模型...")
    # 使用更小的模型配置以适应内存限制
    model = TransformerSummarizer(
        vocab_size=vocab_size,  # 使用实际的词汇表大小
        d_model=512,  # 减小模型维度
        nhead=8,
        num_encoder_layers=6,  # 减少层数
        num_decoder_layers=6,  # 减少层数
        dim_feedforward=1024,  # 减小前馈网络维度
        max_seq_length=max_len,  # 使用实际的最大长度
        dropout=0.2
    )
    model.to(device)
    
    # 初始化训练器
    trainer = ModelTrainer(model, None)
    
    # 执行训练
    print("开始训练模型...")
    trainer.train(train_dataloader, val_dataloader, epochs, learning_rate, device)
    

def predict_summary(args):
    """
    生成摘要
    
    Args:
        args: 命令行参数
    """
    # 初始化数据处理器
    data_processor = CNewsDataProcessor(args.data_path)
    
    # 加载数据
    data = data_processor.load_data()
    
    # 初始化模型
    model = TransformerSummarizer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=512
    )
    
    # 加载预训练模型
    if args.model_path:
        # 加载模型权重
        pass
    
    # 生成摘要
    print("生成摘要...")
    pass

if __name__ == "__main__":
    main()