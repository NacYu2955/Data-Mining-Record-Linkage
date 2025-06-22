import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Evaluator:
    def __init__(self):
        pass
        
    def evaluate(self, true_labels, predicted_labels, total_comparisons, blocked_comparisons):
        """计算评估指标"""
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
        metrics['precision'] = precision_score(true_labels, predicted_labels, zero_division=0)
        metrics['recall'] = recall_score(true_labels, predicted_labels, zero_division=0)
        metrics['f1'] = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # 混淆矩阵相关指标
        cm = confusion_matrix(true_labels, predicted_labels)
        if cm.size == 1:  # 如果只有一个类别
            tn, fp, fn, tp = 0, 0, 0, cm[0][0]
        else:
            tn, fp, fn, tp = cm.ravel()
            
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        metrics['confusion_matrix'] = cm
        
        # 领域特定指标
        metrics['reduction_ratio'] = 1 - (blocked_comparisons / total_comparisons) if total_comparisons > 0 else 0
        metrics['pairs_completeness'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['pairs_quality'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # ROC和PR曲线相关指标
        if len(np.unique(true_labels)) > 1:  # 确保有多个类别
            try:
                metrics['roc_auc'] = roc_auc_score(true_labels, predicted_labels)
                metrics['average_precision'] = average_precision_score(true_labels, predicted_labels)
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
                metrics['roc_curve'] = (fpr, tpr)
                # 计算PR曲线
                precision, recall, _ = precision_recall_curve(true_labels, predicted_labels)
                metrics['pr_curve'] = (precision, recall)
            except:
                metrics['roc_auc'] = 0
                metrics['average_precision'] = 0
                metrics['roc_curve'] = None
                metrics['pr_curve'] = None
        else:
            metrics['roc_auc'] = 0
            metrics['average_precision'] = 0
            metrics['roc_curve'] = None
            metrics['pr_curve'] = None
        
        return metrics



def evaluate_mappings():
    """评估映射结果"""
    try:
        # 读取映射结果
        mappings_df = pd.read_csv('data/mappings_sheet7_all.csv')
            
        # 计算总比较次数和分块后的比较次数
        total_comparisons = len(mappings_df)
        blocked_comparisons = len(mappings_df)
        
        # 使用ID判断是否为真匹配
        true_labels = (mappings_df['test_id'] == mappings_df['ref_id']).astype(int)
        predicted_labels = (mappings_df['similarity'] >= 0.6).astype(int)
        
        # 计算评估指标
        evaluator = Evaluator()
        metrics = evaluator.evaluate(true_labels, predicted_labels, total_comparisons, blocked_comparisons)
        
        print("\n=== 映射结果评估 ===")
        print(f"映射结果总数: {len(mappings_df)}")
        print(f"真实匹配数: {sum(true_labels)}")
        print(f"预测匹配数: {sum(predicted_labels)}")
        
        # 输出详细评估指标
        print("\n=== 详细评估指标 ===")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        
        print("\n=== 混淆矩阵 ===")
        print(f"真正例 (TP): {metrics['true_positives']}")
        print(f"假正例 (FP): {metrics['false_positives']}")
        print(f"真负例 (TN): {metrics['true_negatives']}")
        print(f"假负例 (FN): {metrics['false_negatives']}")
        
        # 输出详细的匹配结果
        print("\n=== 匹配结果统计 ===")
        correct_matches = 0
        incorrect_matches = 0
        
        for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
            if pred == 1:  # 只显示预测为匹配的结果
                row = mappings_df.iloc[i]
                status = "✓" if true == 1 else "✗"
                if true == 1:
                    correct_matches += 1
                else:
                    incorrect_matches += 1
        
        print(f"正确匹配数: {correct_matches}")
        print(f"错误匹配数: {incorrect_matches}")
        print(f"正确匹配率: {correct_matches/(correct_matches+incorrect_matches):.4f}")
        

    except FileNotFoundError as e:
        print(f"错误：找不到必要的文件 - {str(e)}")
    except Exception as e:
        print(f"评估映射结果时发生错误：{str(e)}")

if __name__ == "__main__":
    evaluate_mappings() 