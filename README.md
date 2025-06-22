# 数据挖掘项目：姓名匹配与去重系统

## 项目概述

这是一个基于深度学习和传统算法的姓名匹配与去重系统，主要用于处理大规模姓名数据集的相似度计算、去重和映射任务。项目支持多种数据处理方式，包括Jaccard n-gram相似度和BERT语义相似度。

## 功能特性

- **多模式相似度计算**：支持Jaccard n-gram和BERT语义相似度
- **GPU加速处理**：利用PyTorch和CUDA进行大规模数据加速处理
- **智能数据预处理**：包含多种文本标准化和清洗功能
- **分块算法**：使用FAISS进行高效的数据分块和索引
- **批量处理**：支持大规模数据的分批处理
- **综合评估指标**：提供准确率、精确率、召回率、F1分数等评估指标

## 项目结构

```
HW1/
├── main.py                 # 主程序入口
├── preprocessing.py        # 数据预处理模块
├── similarity.py          # 相似度计算模块
├── blocking.py            # 分块算法模块
├── evaluation.py          # 评估指标计算模块
├── model.py               # 模型定义
├── requirements.txt       # 依赖包列表
├── data/                  # 数据文件夹
│   ├── primary.csv        # 主要数据集
│   ├── alternate.csv      # 备用数据集
│   ├── sheet1-8.csv       # 测试数据集（8种不同处理方式）
│   ├── mappings_*.csv     # 映射结果文件
│   └── duplicates.csv     # 去重结果文件
├── bert-base-uncased/     # BERT预训练模型
└── __pycache__/          # Python缓存文件
```

## 数据处理方式

项目支持8种不同的数据处理方式：

| Sheet | 处理方式 | 描述 |
|-------|----------|------|
| Sheet1 | 去除特殊字符和空格 | 标准化文本格式 |
| Sheet2 | 打乱单词顺序+去除特殊字符 | 测试词序变化的影响 |
| Sheet3 | 字符间空格分隔 | 模拟OCR识别结果 |
| Sheet4 | 打乱单词顺序+字符分隔 | 组合处理方式 |
| Sheet5 | 单词移除 | 模拟数据缺失 |
| Sheet6 | 单词截断 | 模拟数据截断 |
| Sheet7 | 首字母缩写 | 信息压缩处理 |
| Sheet8 | 模拟姓名 | 无真实匹配的测试数据 |

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖包

- `torch>=1.8.0` - PyTorch深度学习框架
- `transformers>=4.0.0` - Hugging Face转换器库
- `pandas>=1.2.0` - 数据处理
- `numpy>=1.19.2` - 数值计算
- `scikit-learn>=0.24.0` - 机器学习工具
- `faiss-cpu>=1.7.0` - 向量索引和搜索
- `sentence-transformers==2.2.0` - 句子嵌入
- `spacy==2.3.7` - 自然语言处理

## BERT模型下载

项目使用BERT-base-uncased预训练模型。首次运行时会自动下载，也可以手动下载：

### 方法1：自动下载（推荐）
首次运行程序时会自动下载BERT模型到本地缓存目录。

### 方法2：手动下载
```bash
# 使用Python脚本下载
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"

# 或者使用transformers-cli
transformers-cli download bert-base-uncased
```

### 方法3：离线下载
如果网络环境受限，可以从以下地址手动下载：
- 模型文件：https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
- 配置文件：https://huggingface.co/bert-base-uncased/resolve/main/config.json
- 词汇表：https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

下载后将文件放置在 `bert-base-uncased/` 目录下。

## 使用方法

### 1. 基本运行

```bash
python main.py
```

### 2. 数据预处理测试

```bash
python preprocessing_test.py
```

### 3. 批量评估

```bash
python batch_evaluate.py
```

### 4. 单独评估

```bash
python evaluation.py
```

## 核心模块说明

### 1. 数据预处理 (preprocessing.py)

提供多种文本标准化功能：
- 特殊字符去除
- 大小写转换
- 昵称映射
- 拼写变体处理
- 数字和罗马数字标准化
- 形近字符处理

### 2. 相似度计算 (similarity.py)

支持两种相似度计算方法：
- **BERT语义相似度**：使用预训练BERT模型计算语义相似度
- **Levenshtein距离**：计算编辑距离相似度
- **综合相似度**：结合多种方法的加权结果

### 3. 分块算法 (blocking.py)

使用FAISS进行高效的数据分块：
- BERT嵌入生成
- 向量索引构建
- 聚类分块
- 相似度搜索

### 4. 评估模块 (evaluation.py)

提供全面的评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- 混淆矩阵
- ROC曲线和PR曲线

## 性能特点

### GPU加速
- 支持CUDA加速的相似度计算
- 批量处理减少内存占用
- 自动内存清理机制

### 大规模数据处理
- 分批处理支持超大数据集
- 分块算法减少计算复杂度
- 高效的向量索引和搜索

### 多模式支持
- Jaccard n-gram模式：适合精确匹配
- BERT语义模式：适合模糊匹配
- 可配置的相似度阈值

## 实验结果

根据项目中的评估结果，不同数据处理方式的性能表现：

- **Sheet1-4**：表现较好，准确率在0.8以上
- **Sheet5-6**：中等表现，准确率在0.6-0.8之间
- **Sheet7**：表现较差，准确率约0.58，主要因为首字母缩写信息损失严重
- **Sheet8**：模拟数据，无真实匹配

## 配置参数

主要可配置参数：

```python
# 相似度阈值
similarity_threshold = 0.6

# 批处理大小
batch_size = 1000

# n-gram大小
n_gram_size = 3

# 是否使用极端预处理
use_extreme_preprocessing = False
```

## 注意事项

1. **GPU内存**：处理大规模数据时注意GPU内存使用
2. **模型下载**：首次运行会自动下载BERT模型
3. **数据格式**：确保输入数据包含ID和NAME列
4. **文件路径**：确保数据文件路径正确

## 扩展功能

项目支持以下扩展：
- 自定义预处理规则
- 新的相似度算法
- 不同的分块策略
- 可视化结果展示

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请联系项目维护者。


 