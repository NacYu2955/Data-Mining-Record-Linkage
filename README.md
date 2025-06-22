# Data Mining Record Linkage

## Project Overview

This is a name matching and deduplication system based on deep learning and traditional algorithms, primarily used for similarity calculation, deduplication, and mapping tasks on large-scale name datasets. The project supports multiple data processing methods, including Jaccard n-gram similarity and BERT semantic similarity.

## Features

- **Multi-mode Similarity Calculation**: Supports Jaccard n-gram and BERT semantic similarity
- **GPU Accelerated Processing**: Utilizes PyTorch and CUDA for large-scale data acceleration
- **Intelligent Data Preprocessing**: Includes various text standardization and cleaning functions
- **Blocking Algorithm**: Uses FAISS for efficient data blocking and indexing
- **Batch Processing**: Supports batch processing of large-scale data
- **Comprehensive Evaluation Metrics**: Provides accuracy, precision, recall, F1-score and other evaluation metrics

## Project Structure

```
/
├── main.py                 # Main program entry
├── preprocessing.py        # Data preprocessing module
├── similarity.py           # Similarity calculation module
├── blocking.py             # Blocking algorithm module
├── evaluation.py           # Evaluation metrics calculation module
├── model.py                # Model definition
├── requirements.txt        # Dependency package list
├── data/                   # Data folder
│   ├── primary.csv         # Primary dataset
│   ├── alternate.csv       # Alternate dataset
│   ├── sheet1-8.csv        # Test datasets (8 different processing methods)
│   ├── mappings_*.csv      # Mapping result files
│   └── duplicates.csv      # Deduplication result files
├── bert-base-uncased/      # BERT pre-trained model
└── __pycache__/            # Python cache files
```

## Data Processing Methods

The project supports 8 different data processing methods:

| Sheet | Processing Method | Description |
|-------|------------------|-------------|
| Sheet1 | Remove special characters and spaces | Standardize text format |
| Sheet2 | Shuffle word order + remove special characters | Test the impact of word order changes |
| Sheet3 | Space-separated characters | Simulate OCR recognition results |
| Sheet4 | Shuffle word order + character separation | Combined processing method |
| Sheet5 | Word removal | Simulate data missing |
| Sheet6 | Word truncation | Simulate data truncation |
| Sheet7 | Initials abbreviation | Information compression processing |
| Sheet8 | Simulated names | Test data without real matches |

## Installation

```bash
pip install -r requirements.txt
```

### Main Dependencies

- `torch>=1.8.0` 
- `transformers>=4.0.0` 
- `pandas>=1.2.0` 
- `numpy>=1.19.2` 
- `scikit-learn>=0.24.0` 
- `faiss-cpu>=1.7.0` 
- `sentence-transformers==2.2.0` 
- `spacy==2.3.7` 

## BERT Model Download

The project uses the BERT-base-uncased pre-trained model. It will be automatically downloaded on first run, or you can download it manually:

### Method 1: Automatic Download (Recommended)
The BERT model will be automatically downloaded to the local cache directory on first run.

### Method 2: Manual Download
```bash
# Download using Python script
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"

# Or use transformers-cli
transformers-cli download bert-base-uncased
```

### Method 3: Offline Download
If network access is restricted, you can manually download from the following addresses:
- Model file: https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
- Config file: https://huggingface.co/bert-base-uncased/resolve/main/config.json
- Vocabulary: https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

Place the downloaded files in the `bert-base-uncased/` directory.

## Usage

### 1. Basic Run

```bash
python main.py
```

### 2. Data Preprocessing Test

```bash
python preprocessing_test.py
```

### 3. Batch Evaluation

```bash
python batch_evaluate.py
```

### 4. Individual Evaluation

```bash
python evaluation.py
```

## Core Modules

### 1. Data Preprocessing (preprocessing.py)

Provides various text standardization functions:
- Special character removal
- Case conversion
- Nickname mapping
- Spelling variant processing
- Number and Roman numeral standardization
- Similar character processing

### 2. Similarity Calculation (similarity.py)

Supports two similarity calculation methods:
- **BERT Semantic Similarity**: Uses pre-trained BERT model to calculate semantic similarity
- **Levenshtein Distance**: Calculates edit distance similarity
- **Combined Similarity**: Weighted combination of multiple methods

### 3. Blocking Algorithm (blocking.py)

Uses FAISS for efficient data blocking:
- BERT embedding generation
- Vector index construction
- Clustering blocking
- Similarity search

### 4. Evaluation Module (evaluation.py)

Provides comprehensive evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and PR curve

## Performance Features

### GPU Acceleration
- Supports CUDA-accelerated similarity calculation
- Batch processing reduces memory usage
- Automatic memory cleanup mechanism

### Large-scale Data Processing
- Batch processing supports ultra-large datasets
- Blocking algorithm reduces computational complexity
- Efficient vector indexing and search

### Multi-mode Support
- Jaccard n-gram mode: Suitable for exact matching
- BERT semantic mode: Suitable for fuzzy matching
- Configurable similarity thresholds

## Experimental Results

| Test      | Total number of mapping results | Actual matching number | Accuracy | Precision | Recall | F1     |
|-----------|---------------------------------|------------------------|----------|-----------|--------|--------|
| test_01   | 16032                           | 15185                  | 0.9472   | 1         | 0.9472 | 0.9729 |
| sheet1    | 1000                            | 997                    | 0.997    | 0.997     | 1      | 0.9985 |
| sheet2    | 975                             | 957                    | 0.7231   | 0.9831    | 0.7304 | 0.8381 |
| sheet3    | 1000                            | 995                    | 0.995    | 0.995     | 1      | 0.9975 |
| sheet4    | 980                             | 967                    | 0.7245   | 0.9874    | 0.7301 | 0.8395 |
| sheet5    | 967                             | 915                    | 0.7622   | 0.9647    | 0.777  | 0.8608 |
| sheet6    | 964                             | 943                    | 0.8133   | 0.9848    | 0.8218 | 0.896  |
| sheet7    | 707                             | 648                    | 0.5813   | 0.9706    | 0.5602 | 0.7104 |
| sheet8    | 0                               | 0                      | -        | -         | -      | -      |

## Configuration Parameters

Main configurable parameters:

```python
# Similarity threshold
similarity_threshold = 0.6

# Batch size
batch_size = 1000

# n-gram size
n_gram_size = 3

# Whether to use extreme preprocessing
use_extreme_preprocessing = False
```

## Notes

1. **GPU Memory**: Pay attention to GPU memory usage when processing large-scale data
2. **Model Download**: BERT model will be automatically downloaded on first run
3. **Data Format**: Ensure input data contains ID and NAME columns
4. **File Paths**: Ensure data file paths are correct

## Extensions

The project supports the following extensions:
- Custom preprocessing rules
- New similarity algorithms
- Different blocking strategies
- Visualization of results






 