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
    epochs = 20
    batch_size = 32
    max_len = 1024
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备: ", device)

    train_model(train_data_path, test_data_path, val_data_path, max_len, epochs, batch_size, learning_rate, train_tgt_path, val_tgt_path)

def train_model(train_data_path, test_data_path, val_data_path, max_len, epochs, batch_size, learning_rate, train_tgt_path, val_tgt_path):
    """
    训练模型
    
    Args:
        args: 命令行参数
    """
    # 初始化数据处理器
    data_processor = CNewsDataProcessor(train_data_path, val_data_path, train_tgt_path,val_tgt_path, max_len)
    
    # 加载和预处理数据
    train_data, val_data = data_processor.load_data()

    train_dataset = TextSummaryDataset(train_data, jieba, max_len)
    val_dataset = TextSummaryDataset(val_data, jieba, max_len)



    #创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print("使用设备: ", device)
    # 初始化模型
    model = TransformerSummarizer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=1024,
        dropout=0.2
    )
    model.to(device)
    
    # 初始化训练器
    trainer = ModelTrainer(model, None)
    
    # 执行训练
    print("开始训练模型...")
    trainer.train(train_dataloader, val_dataloader, epochs, learning_rate, 'cpu')

    

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