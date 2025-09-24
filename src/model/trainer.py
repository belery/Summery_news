import torch
from utils.helpers import decode_summary, get_pad_mask
class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(self, model, tokenizer):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            tokenizer: 分词器
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def train(self, train_dataloader, val_dataloader, epochs, learning_rate, device):
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            device: 训练设备
        """
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct, total = 0, 0
            for batch_idx, batch in enumerate(train_dataloader):
                src, tgt = batch
                src = src.to(device)
                tgt = tgt.to(device)

                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = get_pad_mask(src, tgt, pad_token_id=0)
                tgt_seq_len = decoder_input.size(1)
                tgt_caueal_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1)
                tgt_caueal_mask = tgt_caueal_mask.masked_fill(tgt_caueal_mask == 1, float('-inf')).to(device)


                #前向传播
                output = self.model(
                    src, decoder_input
                    , src_mask, tgt_mask=tgt_caueal_mask, 
                    src_key_padding_mask=src_key_padding_mask, 
                    tgt_key_padding_mask=tgt_key_padding_mask)
                loss = criterion(output.view(-1, output.shape[-1]), tgt[1:].view(-1))
                #反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(dim=-1)
                total += (predicted == tgt[1:]).sum().item()
                correct += (predicted == tgt[1:]).sum().item()

                if batch_idx % 100 == 0:
                    acc = 100 * correct / total
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
            
            avg_loss = total_loss / len(train_dataloader)
            acc =  100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}], Avg. Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

                
                
    
    def evaluate(self, dataloader, device):
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            device: 评估设备
            
        Returns:
            dict: 评估结果
        """
        pass
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        pass
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path (str): 模型路径
        """
        pass