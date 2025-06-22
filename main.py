import pandas as pd
import numpy as np
from blocking import Blocker
from similarity import SimilarityCalculator
from preprocessing import Preprocessor
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def deduplicate_records(data: pd.DataFrame, similarity_threshold: float = 0.8, use_extreme_preprocessing: bool = False) -> tuple:
    """使用GPU加速的Jaccard n-gram相似度进行去重处理（分批处理）
    返回：(重复记录列表, 去重后的数据DataFrame)"""
    print("正在创建n-gram特征矩阵...")
    # 创建特征矩阵
    matrix, ngram_to_idx = create_ngram_matrix(data['NAME'].tolist(), n=3)
    
    print("正在分批计算相似度...")
    duplicates = []
    duplicate_indices = set()  # 用于记录重复记录的索引
    batch_size = 1000  # 每批处理的记录数
    
    # 分批处理
    for i in tqdm(range(0, len(data), batch_size), desc="处理批次"):
        end_idx = min(i + batch_size, len(data))
        current_batch = matrix[i:end_idx]
        
        # 计算当前批次与所有记录的相似度
        intersection = torch.mm(current_batch, matrix.t())
        sum_batch = current_batch.sum(dim=1, keepdim=True)
        sum_all = matrix.sum(dim=1, keepdim=True)
        union = sum_batch + sum_all.t() - intersection
        
        # 计算Jaccard相似度
        similarity_matrix = intersection / (union + 1e-8)
        
        # 只保留上三角矩阵部分（避免重复比较）
        for j in range(len(current_batch)):
            row_idx = i + j
            # 只比较当前行之后的数据
            similarities = similarity_matrix[j, row_idx+1:]
            indices = torch.where(similarities >= similarity_threshold)[0]
            
            for idx in indices:
                col_idx = (row_idx + 1 + idx).item()
                similarity = similarities[idx].item()
                duplicates.append({
                    'id1': data.iloc[row_idx]['ID'],
                    'id2': data.iloc[col_idx]['ID'],
                    'name1': data.iloc[row_idx]['NAME'],
                    'name2': data.iloc[col_idx]['NAME'],
                    'similarity': similarity
                })
                # 记录重复记录的索引
                duplicate_indices.add(row_idx)
                duplicate_indices.add(col_idx)
        
        # 清理GPU内存
        del intersection, sum_batch, sum_all, union, similarity_matrix
        torch.cuda.empty_cache()
    
    # 生成去重后的数据
    deduplicated_data = data[~data.index.isin(duplicate_indices)].copy()
    
    return duplicates, deduplicated_data

def ngram_set(s, n=3):
    return set([s[i:i+n] for i in range(len(s)-n+1)]) if len(s) >= n else set([s])

def jaccard_sim(s1, s2, n=3):
    set1 = ngram_set(s1, n)
    set2 = ngram_set(s2, n)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def create_ngram_matrix(texts, n=3, ngram_to_idx=None):
    """创建n-gram特征矩阵"""
    if ngram_to_idx is None:
        # 如果是第一次调用，创建新的特征空间
        all_ngrams = set()
        for text in texts:
            all_ngrams.update(ngram_set(text, n))
        ngram_to_idx = {ngram: idx for idx, ngram in enumerate(all_ngrams)}
    
    matrix = torch.zeros((len(texts), len(ngram_to_idx)), device=device)
    
    for i, text in enumerate(texts):
        ngrams = ngram_set(text, n)
        for ngram in ngrams:
            if ngram in ngram_to_idx:
                matrix[i, ngram_to_idx[ngram]] = 1
    
    return matrix, ngram_to_idx

def map_test_records_jaccard(test_data, reference_data, threshold=0.4, n=3, return_all_best=False):
    """使用GPU加速的Jaccard n-gram相似度进行全量比对映射
    return_all_best: 是否返回所有test_id的最佳映射（不管阈值）
    """
    print("正在创建n-gram特征矩阵...")
    # 首先使用参考数据创建特征空间
    ref_matrix, ngram_to_idx = create_ngram_matrix(reference_data['NAME'].tolist(), n)
    # 然后使用相同的特征空间创建测试数据矩阵
    test_matrix, _ = create_ngram_matrix(test_data['NAME'].tolist(), n, ngram_to_idx)
    
    print("正在计算相似度矩阵...")
    # 计算交集和并集
    intersection = torch.mm(test_matrix, ref_matrix.t())
    test_sum = test_matrix.sum(dim=1, keepdim=True)
    ref_sum = ref_matrix.sum(dim=1, keepdim=True)
    union = test_sum + ref_sum.t() - intersection
    
    # 计算Jaccard相似度
    similarity_matrix = intersection / (union + 1e-8)  # 添加小量避免除零
    
    print("正在生成映射结果...")
    mappings = []
    all_best_mappings = []
    for i in range(len(test_data)):
        best_sim, best_idx = similarity_matrix[i].max(dim=0)
        best_sim = best_sim.item()
        best_idx = best_idx.item()
        
        # 全量映射（每个test_id都保留）
        all_best_mappings.append({
            'test_id': test_data.iloc[i]['ID'],
            'ref_id': reference_data.iloc[best_idx]['ID'],
            'test_name': test_data.iloc[i]['NAME'],
            'ref_name': reference_data.iloc[best_idx]['NAME'],
            'similarity': best_sim
        })
        # 阈值筛选的映射
        if best_sim >= threshold:
            mappings.append({
                'test_id': test_data.iloc[i]['ID'],
                'ref_id': reference_data.iloc[best_idx]['ID'],
                'test_name': test_data.iloc[i]['NAME'],
                'ref_name': reference_data.iloc[best_idx]['NAME'],
                'similarity': best_sim
            })
    
    if return_all_best:
        return mappings, all_best_mappings
    else:
        return mappings

def main():
    # 选择工作流模式：'jaccard' 或 'semantic'
    mode = 'jaccard'  # 可改为'semantic'以使用分块+语义相似度

    print(f"当前工作流模式: {mode}")
    # 读取数据文件
    print("正在读取数据文件...")
    primary_df = pd.read_csv('data/primary.csv')
    alternate_df = pd.read_csv('data/alternate.csv')
    test_df = pd.read_csv('data/sheet7.csv')
    
    # 检查并重命名test_01.csv的列名
    if 'VARIANT' in test_df.columns:
        test_df = test_df.rename(columns={'VARIANT': 'NAME'})
    
    pre = Preprocessor()
    
    if mode == 'jaccard':
        # 对所有数据做普通预处理
        primary_df['NAME'] = primary_df['NAME'].apply(pre.preprocess)
        alternate_df['NAME'] = alternate_df['NAME'].apply(pre.preprocess)
        test_df['NAME'] = test_df['NAME'].apply(pre.preprocess)
        
        # 合并primary和alternate数据
        all_data = pd.concat([primary_df, alternate_df], ignore_index=True)
        print(f"\n合并后总记录数: {len(all_data)}")
        
        # 恢复去重逻辑
        use_extreme_preprocessing = False
        similarity_threshold = 0.6
        print("\n开始去重处理...")
        duplicates, deduplicated_data = deduplicate_records(all_data, similarity_threshold, use_extreme_preprocessing)
        
        # 保存去重结果
        duplicates_df = pd.DataFrame(duplicates)
        duplicates_df.to_csv('data/duplicates.csv', index=False)
        print(f"去重结果已保存到 duplicates.csv")
        print(f"找到 {len(duplicates)} 对重复记录")
        
        # 保存去重后的数据
        deduplicated_data.to_csv('data/deduplicated_data.csv', index=False)
        print(f"去重后的数据已保存到 deduplicated_data.csv")
        print(f"去重后剩余 {len(deduplicated_data)} 条记录")
        
        # Jaccard n-gram全量比对映射
        print("\n开始Jaccard n-gram映射...")
        mappings, all_best_mappings = map_test_records_jaccard(test_df, deduplicated_data, threshold=0.6, n=3, return_all_best=True)
        
        # 保存映射结果
        mappings_df = pd.DataFrame(mappings)
        mappings_df.to_csv('data/mappings_sheet7.csv', index=False)
        print(f"映射结果已保存到 mappings_sheet7.csv")
        print(f"找到 {len(mappings)} 个匹配")
        # 保存全量映射结果
        all_best_mappings_df = pd.DataFrame(all_best_mappings)
        all_best_mappings_df.to_csv('data/mappings_sheet7_all.csv', index=False)
        print(f"全量映射结果已保存到 mappings_sheet7_all.csv")
        print(f"全量映射包含 {len(all_best_mappings)} 个test样本")
    
    elif mode == 'semantic':
        # 预处理
        primary_df['NAME'] = primary_df['NAME'].apply(pre.preprocess)
        alternate_df['NAME'] = alternate_df['NAME'].apply(pre.preprocess)
        test_df['NAME'] = test_df['NAME'].apply(pre.preprocess)
        
        # 合并primary和alternate数据
        all_data = pd.concat([primary_df, alternate_df], ignore_index=True)
        print(f"\n合并后总记录数: {len(all_data)}")
        
        # 恢复去重逻辑
        use_extreme_preprocessing = False
        similarity_threshold = 0.6
        print("\n开始去重处理...")
        duplicates, deduplicated_data = deduplicate_records(all_data, similarity_threshold, use_extreme_preprocessing)
        
        # 保存去重结果
        duplicates_df = pd.DataFrame(duplicates)
        duplicates_df.to_csv('data/duplicates.csv', index=False)
        print(f"去重结果已保存到 duplicates.csv")
        print(f"找到 {len(duplicates)} 对重复记录")
        
        # 保存去重后的数据
        deduplicated_data.to_csv('data/deduplicated_data.csv', index=False)
        print(f"去重后的数据已保存到 deduplicated_data.csv")
        print(f"去重后剩余 {len(deduplicated_data)} 条记录")
        
        # 1. Blocking
        print("\n开始分块...")
        blocker = Blocker()
        all_embeddings = blocker.generate_embeddings(deduplicated_data['NAME'].tolist())
        test_embeddings = blocker.generate_embeddings(test_df['NAME'].tolist())
        all_blocks = blocker.blocking(all_embeddings)
        test_blocks = blocker.blocking(test_embeddings)
        
        # 2. Similarity Computation
        print("\n开始相似度计算...")
        similarity_calculator = SimilarityCalculator()
        mappings = []
        all_best_mappings = []
        for i, test_row in test_df.iterrows():
            test_name = test_row['NAME']
            test_id = test_row['ID']
            # 找到与test样本同block的候选ref
            candidate_indices = np.where(all_blocks == test_blocks[i])[0]
            best_sim = -1
            best_idx = None
            for idx in candidate_indices:
                ref_name = deduplicated_data.iloc[idx]['NAME']
                ref_id = deduplicated_data.iloc[idx]['ID']
                sim = similarity_calculator.calculate_similarity(test_name, ref_name)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            if best_idx is not None:
                # 全量映射
                all_best_mappings.append({
                    'test_id': test_id,
                    'ref_id': deduplicated_data.iloc[best_idx]['ID'],
                    'test_name': test_name,
                    'ref_name': deduplicated_data.iloc[best_idx]['NAME'],
                    'similarity': best_sim
                })
                # 阈值筛选
                if best_sim >= 0.6:
                    mappings.append({
                        'test_id': test_id,
                        'ref_id': deduplicated_data.iloc[best_idx]['ID'],
                        'test_name': test_name,
                        'ref_name': deduplicated_data.iloc[best_idx]['NAME'],
                        'similarity': best_sim
                    })
        # 保存映射结果
        mappings_df = pd.DataFrame(mappings)
        mappings_df.to_csv('data/mappings_sheet7.csv', index=False)
        print(f"映射结果已保存到 mappings_sheet7.csv")
        print(f"找到 {len(mappings)} 个匹配")
        # 保存全量映射结果
        all_best_mappings_df = pd.DataFrame(all_best_mappings)
        all_best_mappings_df.to_csv('data/mappings_sheet7_all.csv', index=False)
        print(f"全量映射结果已保存到 mappings_sheet7_all.csv")
        print(f"全量映射包含 {len(all_best_mappings)} 个test样本")
    else:
        print("未知模式，请选择 'jaccard' 或 'semantic'")

if __name__ == "__main__":
    main() 