import pandas as pd
import numpy as np
from main import map_test_records
from evaluation import Evaluator, print_metrics

# 读取参考数据
primary_df = pd.read_csv('data/primary.csv')
alternate_df = pd.read_csv('data/alternate.csv')
reference_data = pd.concat([primary_df, alternate_df], ignore_index=True)

# 读取Excel文件
excel_path = 'data/test_02.xlsx'
sheet_names = [f'Sheet{i}' for i in range(1, 8)]

for sheet in sheet_names:
    print(f"\n===== 正在评估 {sheet} =====")
    # 读取当前Sheet
    test_df = pd.read_excel(excel_path, sheet_name=sheet)
    # 保证有ID和NAME列
    if 'VARIANT' in test_df.columns:
        test_df = test_df.rename(columns={'VARIANT': 'NAME'})
    # 运行映射
    mappings = map_test_records(test_df, reference_data)
    mappings_df = pd.DataFrame(mappings)
    mappings_df.to_csv(f'data/mappings_{sheet}.csv', index=False)
    # 评估
    if len(mappings_df) == 0:
        print("无映射结果")
        continue
    true_labels = (mappings_df['test_id'] == mappings_df['ref_id']).astype(int)
    predicted_labels = (mappings_df['similarity'] >= 0.8).astype(int)
    total_comparisons = len(mappings_df) * (len(mappings_df) - 1) // 2
    blocked_comparisons = len(mappings_df)
    evaluator = Evaluator()
    metrics = evaluator.evaluate(true_labels, predicted_labels, total_comparisons, blocked_comparisons)
    print_metrics(metrics)