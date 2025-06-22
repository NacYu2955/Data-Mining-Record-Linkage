import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from preprocessing import Preprocessor
import os

class SimilarityCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_extreme_preprocessing=False):
        self.device = device
        self.preprocessor = Preprocessor()
        self.use_extreme_preprocessing = use_extreme_preprocessing
        
        print("正在加载BERT模型...")
        # 使用预训练模型
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model = self.model.to(device)
        # 设置为评估模式，减少内存使用
        self.model.eval()
        print("BERT模型加载完成！")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算综合相似度分数"""
        # 预处理文本
        if self.use_extreme_preprocessing:
            text1_processed = self.preprocessor.preprocess_extreme(text1)
            text2_processed = self.preprocessor.preprocess_extreme(text2)
        else:
            text1_processed = self.preprocessor.preprocess(text1)
            text2_processed = self.preprocessor.preprocess(text2)
        
        # 使用BERT计算相似度
        with torch.no_grad():
            # 编码文本，限制最大长度
            inputs1 = self.tokenizer(
                text1_processed, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128  # 限制最大长度
            ).to(self.device)
            inputs2 = self.tokenizer(
                text2_processed, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128  # 限制最大长度
            ).to(self.device)
            
            # 获取嵌入
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)
            
            # 计算余弦相似度
            embeddings1 = outputs1.last_hidden_state.mean(dim=1)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1)
            
            bert_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
            
            # 清理内存
            del inputs1, inputs2, outputs1, outputs2, embeddings1, embeddings2
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Levenshtein距离
        lev_dist = self._levenshtein_distance(text1_processed, text2_processed)
        lev_sim = 1 - (lev_dist / max(len(text1_processed), len(text2_processed)))
        
        # 如果使用极端预处理，增加Levenshtein距离的权重
        if self.use_extreme_preprocessing:
            return 0.4 * bert_sim + 0.6 * lev_sim
        else:
            return 0.7 * bert_sim + 0.3 * lev_sim
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算Levenshtein距离（使用CPU版本）"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        # 使用numpy数组进行计算
        previous_row = np.arange(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = np.zeros(len(s2) + 1)
            current_row[0] = i + 1
            
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row[j + 1] = min(insertions, deletions, substitutions)
            
            previous_row = current_row
        
        return previous_row[-1] 