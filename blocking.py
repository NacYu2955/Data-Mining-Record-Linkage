import numpy as np
import torch
import faiss
from transformers import BertTokenizer, BertModel
from preprocessing import Preprocessor
import os
from tqdm import tqdm

class Blocker:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', use_extreme_preprocessing=False):
        self.device = device
        self.preprocessor = Preprocessor()
        self.use_extreme_preprocessing = use_extreme_preprocessing
        
        print("正在加载BERT模型...")
        # 使用预训练模型生成更好的嵌入
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model = self.model.to(device)
        # 设置为评估模式，减少内存使用
        self.model.eval()
        print("BERT模型加载完成！")
        
        # 初始化FAISS索引
        self.index = None
    
    def generate_embeddings(self, texts: list, batch_size=32) -> np.ndarray:
        """使用BERT生成特征向量"""
        if self.use_extreme_preprocessing:
            processed_texts = [self.preprocessor.preprocess_extreme(text) for text in texts]
        else:
            processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # 使用BERT生成嵌入
        embeddings = []
        self.model.eval()
        
        # 使用tqdm显示进度
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="生成特征向量"):
            batch_texts = processed_texts[i:i + batch_size]
            
            with torch.no_grad():
                # 编码文本，限制最大长度
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取嵌入
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
                
                # 清理内存
                del inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合并所有批次的嵌入
        return np.vstack(embeddings)
    
    def blocking(self, embeddings: np.ndarray) -> np.ndarray:
        """使用FAISS进行分块"""
        # 将数据转换为float32
        embeddings = embeddings.astype('float32')
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # 添加向量到索引
        self.index.add(embeddings)
        
        # 使用k-means聚类
        k = min(100, len(embeddings) // 2)  # 设置合适的聚类数
        _, cluster_labels = self.index.search(embeddings, k)
        
        return cluster_labels[:, 0]  # 返回每个点的最近聚类中心 