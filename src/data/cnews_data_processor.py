from collections import Counter
import jieba
class CNewsDataProcessor:
    """
    CNews数据集处理器
    """
    
    def __init__(self, train_data_path, val_data_path,train_tgt_path, val_tgt_path, max_len):
        """
        初始化数据处理器
        
        Args:
            data_path (str): 数据集路径
        """
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.train_tgt_path = train_tgt_path
        self.val_tgt_path = val_tgt_path
        self.max_len = max_len
    
    def load_data(self):
        """
        加载CNews数据集
        
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        with open(self.train_data_path, 'r', encoding='utf-8') as f_src, \
             open(self.train_tgt_path, 'r', encoding='utf-8') as f_tgt:
            train_src_lines = f_src.readlines()
            train_tgt_lines = f_tgt.readlines()
            train_data = list(zip(train_src_lines, train_tgt_lines))
        
        # 加载验证数据
        with open(self.val_data_path, 'r', encoding='utf-8') as f_src, \
             open(self.val_tgt_path, 'r', encoding='utf-8') as f_tgt:
            val_src_lines = f_src.readlines()
            val_tgt_lines = f_tgt.readlines()
            val_data = list(zip(val_src_lines, val_tgt_lines))
        return train_data, val_data

    
    def preprocess_data(self, data):
        """
        预处理数据
        
        Args:
            data: 原始数据
            
        Returns:
            processed_data: 处理后的数据
        """
        for line in data:
            if len(line.split()) > self.max_len:
                head_line = line[:self.max_len/2]
                tail_line = line[self.max_len/2:]
                len = head_line + tail_line
            else:
                line = line + [0] * (self.max_len - len(line))
        
    
    def tokenize_data(self, data, tokenizer):
        """
        对数据进行tokenize处理
        
        Args:
            data: 待处理数据
            tokenizer: 分词器
            
        Returns:
            tokenized_data: 分词后的数据
        """
        tokenizer_data = []
        for line in data:
            line = line.strip()
            tokens = jieba.cut(line)
            tokenizer_data.append(tokens)
        return tokenizer_data
