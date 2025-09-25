import torch
from torch.utils.data import Dataset
import pickle
import os

class TextSummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, vocab_file='vocab.pkl'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_file = vocab_file
        
        # 构建词汇表
        print("开始构建词汇表...")
        self.word2idx, self.idx2word = self._build_vocab()
        print("词汇表构建完成")
    
    def _build_vocab(self):
        if os.path.exists(self.vocab_file):
            print("从缓存文件加载词汇表...")
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
            print(f"词汇表加载完成，词汇量: {len(vocab['word2idx'])}")
            return vocab['word2idx'], vocab['idx2word']
        
        print("从数据构建词汇表...")
        # 构建词汇表
        word_freq = {}
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        # 统计词频
        total_items = len(self.data)
        print_interval = max(1, total_items // 100)  # 每1%显示一次进度
        
        for idx, (src_text, tgt_text) in enumerate(self.data):
            for text in [src_text, tgt_text]:
                tokens = list(self.tokenizer.cut(text.strip()))
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1
            
            # 显示进度
            if (idx + 1) % print_interval == 0 or idx == total_items - 1:
                progress = (idx + 1) / total_items * 100
                print(f"词汇表构建进度: {progress:.1f}% ({idx+1}/{total_items})")
        
        # 构建词汇表，保留高频词
        vocab_size = 10000
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size-4]
        
        # 正确处理special_tokens
        word2idx = {word: idx for idx, word in enumerate(special_tokens)}
        idx2word = {idx: word for idx, word in enumerate(special_tokens)}
        
        for word, _ in sorted_words:
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word
            
        # 保存词汇表
        vocab = {'word2idx': word2idx, 'idx2word': idx2word}
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"词汇表已保存到 {self.vocab_file}，总词汇量: {len(word2idx)}")
            
        return word2idx, idx2word
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 处理源序列
        src_tokens = list(self.tokenizer.cut(src_text.strip()))
        src_ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in src_tokens]
        # 确保总长度不超过max_length
        src_ids = (src_ids[:self.max_length] if len(src_ids) > self.max_length 
                  else src_ids + [self.word2idx['<PAD>']] * (self.max_length - len(src_ids)))
        
        # 处理目标序列
        tgt_tokens = list(self.tokenizer.cut(tgt_text.strip()))
        tgt_ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tgt_tokens]
        tgt_ids = [self.word2idx['<SOS>']] + tgt_ids + [self.word2idx['<EOS>']]  # 添加SOS和EOS标记
        # 确保总长度不超过max_length
        tgt_ids = (tgt_ids[:self.max_length] if len(tgt_ids) > self.max_length 
                  else tgt_ids + [self.word2idx['<PAD>']] * (self.max_length - len(tgt_ids)))
        
        return (torch.tensor(src_ids, dtype=torch.long), 
                torch.tensor(tgt_ids, dtype=torch.long))