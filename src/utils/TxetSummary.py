import torch
from torch.utils.data import Dataset
class TextSummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        #处理源序列
        src_tokens = list(self.tokenizer.cut(src_text.strip()))
        src_ids = [hash(token) % 9996 + 4 for token in src_tokens]
        src_ids = (src_ids[:self.max_length] if len(src_ids) > self.max_length 
                  else src_ids + [0] * (self.max_length - len(src_ids)))
        
        # 处理目标序列
        tgt_tokens = list(self.tokenizer.cut(tgt_text.strip()))
        tgt_ids = [hash(token) % 9996 + 4 for token in tgt_tokens]
        tgt_ids = [2] + tgt_ids + [3]  # 添加SOS和EOS标记
        tgt_ids = (tgt_ids[:self.max_length] if len(tgt_ids) > self.max_length 
                  else tgt_ids + [0] * (self.max_length - len(tgt_ids)))
        
        return (torch.tensor(src_ids, dtype=torch.long), 
                torch.tensor(tgt_ids, dtype=torch.long))