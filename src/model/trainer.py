import torch
from utils.helpers import decode_summary, get_pad_mask

class ModelTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def _create_causal_mask(self, size, device):
        """创建因果掩码（防止解码器关注未来 token）"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def _create_masks(self, src, tgt_input):
        """统一创建所有需要的 masks，并添加调试信息"""
        device = src.device
        
        # Padding masks: True 表示要忽略的位置 (batch_size, seq_len)
        pad_token_id = 0
        src_padding_mask = (src == pad_token_id)
        tgt_padding_mask = (tgt_input == pad_token_id)

        # 因果掩码（仅用于目标序列自注意力）
        tgt_len = tgt_input.size(1)
        causal_mask = self._create_causal_mask(tgt_len, device)

        if causal_mask.size(0) <= 10:  # 避免输出太长
            print(f"[DEBUG] causal_mask:\n{causal_mask.cpu().float()}")
        
        # 注意：mask在transformer_summarizer.py的forward方法中会被转置
        # 因此这里保持(batch_size, seq_len)格式
        masks = {
            'src_key_padding_mask': src_padding_mask,
            'tgt_key_padding_mask': tgt_padding_mask,
            'memory_key_padding_mask': src_padding_mask,
            'tgt_mask': causal_mask
        }
        
        return masks
    
    def train(self, train_dataloader, val_dataloader, epochs, learning_rate, device):
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

                # decoder_input 是去结尾的 target
                decoder_input = tgt[:, :-1]
                # decoder_target 是去开头的真实标签
                decoder_target = tgt[:, 1:]
                
                # 使用统一函数创建所有 masks
                masks = self._create_masks(src, decoder_input)
                
                
                # 前向传播
                output = self.model(
                    src, 
                    decoder_input,
                    src_mask=None,
                    **masks
                )
                
                # 损失计算
                loss = criterion(output.contiguous().view(-1, output.shape[-1]), 
                               decoder_target.contiguous().view(-1))
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(dim=-1)
                non_pad_mask = decoder_target != 0
                total += non_pad_mask.sum().item()
                correct += ((predicted == decoder_target) & non_pad_mask).sum().item()

                if batch_idx % 100 == 0:
                    acc = 100 * correct / total if total > 0 else 0
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%")
            
            avg_loss = total_loss / len(train_dataloader)
            acc = 100 * correct / total if total > 0 else 0
            print(f"Epoch [{epoch+1}/{epochs}], Avg. Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
            
            val_metrics = self.evaluate(val_dataloader, device)
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            if (epoch + 1) % 5 == 0:
                self.save_model(f"model_epoch_{epoch+1}.pt")
    
    def evaluate(self, dataloader, device):
        self.model.eval()
        total_loss = 0
        correct, total = 0, 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for batch in dataloader:
                src, tgt = batch
                src = src.to(device)
                tgt = tgt.to(device)
                
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                
                # 统一使用 _create_masks
                masks = self._create_masks(src, decoder_input)
                
                output = self.model(
                    src, 
                    decoder_input,
                    src_mask=None,
                    **masks
                )
                
                loss = criterion(output.contiguous().view(-1, output.shape[-1]), 
                               decoder_target.contiguous().view(-1))
                total_loss += loss.item()
                
                _, predicted = output.max(dim=-1)
                non_pad_mask = decoder_target != 0
                total += non_pad_mask.sum().item()
                correct += ((predicted == decoder_target) & non_pad_mask).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def save_model(self, path):
        """
        保存模型
        """
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """
        加载模型权重
        """
        state_dict = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"模型已从 {path} 加载")