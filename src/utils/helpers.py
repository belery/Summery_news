import jieba
from collections import Counter
def create_masks(src, tgt, pad_token_id):
    """
    创建源序列和目标序列的掩码
    
    Args:
        src (torch.Tensor): 源序列
        tgt (torch.Tensor): 目标序列
        pad_token_id (int): 填充标记ID
        
    Returns:
        tuple: (src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
    """
    pass

def calculate_rouge_scores(generated_summaries, reference_summaries):
    """
    计算ROUGE分数
    
    Args:
        generated_summaries (list): 生成的摘要列表
        reference_summaries (list): 参考摘要列表
        
    Returns:
        dict: ROUGE分数
    """
    pass

def decode_summary(tokenizer, summary_tokens):
    """
    解码摘要
    
    Args:
        tokenizer: 分词器
        summary_tokens (torch.Tensor): 摘要标记
        
    Returns:
        str: 解码后的摘要文本
    """
    pass

class Tokenizer:
    """
    分词器
    """
    def __init__(self, vocab_size = 30000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.scial_tokens = {
            '[pad]':0,
            '[unk]':1,
            '[sos]':2,
            '[eos]':3
        }

    def trina_tokenizer(self, texts, use_hmm = True):
        words = Counter()

        for text in texts:
            tokens = jieba.cut(text, HMM=use_hmm)
            words.update(tokens)
        most_word = words.most_common(self.vocab_size - len(self.scial_tokens))

        for key, value in self.scial_tokens.items():
            self.vocab[key] = value

        for word, _ in most_word:
            self.vocab[word] = len(self.vocab)

        self.inverse_vocab = {value:key for key, value in self.vocab.items()}
        return self.vocab_size, self.inverse_vocab ,self.scial_tokens, self.vocab
    
    def tokenize(self, text):
        tokenized_text = [self.scial_tokens['[sos]']] + list(jieba.cut(text)) + [self.scial_tokens['[eos]']]
        return tokenized_text
    
if __name__ == '__main__':
    tokenizer = Tokenizer()
    tokenizer.trina_tokenizer(['今天天气不错'])
    print(tokenizer.tokenize('今天天气不错'))