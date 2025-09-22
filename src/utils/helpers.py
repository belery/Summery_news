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