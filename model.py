import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import Preprocessor

class RecordMatcher:
    def __init__(self):
        self.model = LogisticRegression()
        self.vectorizer = TfidfVectorizer()
        self.preprocessor = Preprocessor()
    
    def train(self, train_data, epochs: int = 10):
        """训练模型"""
        # 准备特征
        X = []
        y = []
        
        for _, row in train_data.iterrows():
            text1 = self.preprocessor.preprocess(row['text1'])
            text2 = self.preprocessor.preprocess(row['text2'])
            
            # 组合两个文本的特征
            combined_text = f"{text1} {text2}"
            X.append(combined_text)
            y.append(row['label'])
        
        # 向量化
        X = self.vectorizer.fit_transform(X)
        
        # 训练模型
        self.model.fit(X, y)
    
    def predict(self, text1: str, text2: str) -> float:
        """预测两个文本的相似度"""
        text1 = self.preprocessor.preprocess(text1)
        text2 = self.preprocessor.preprocess(text2)
        combined_text = f"{text1} {text2}"
        
        X = self.vectorizer.transform([combined_text])
        return self.model.predict_proba(X)[0][1] 