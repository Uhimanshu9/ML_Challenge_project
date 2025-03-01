# Entity Extraction from Product Images: Comparing NLP and Rule-Based Approaches

## Overview
This repository contains code and resources for the research paper "Comparative Analysis of NLP and Rule-Based Approaches for Entity Extraction from Product Images Using OCR." The project evaluates different methods for extracting product metadata (weights, dimensions, voltage, etc.) from product images using Optical Character Recognition (OCR) and subsequent entity extraction techniques.

## Research Summary
The study compares traditional rule-based (regular expression) approaches with advanced NLP models (BERT, T5) for extracting structured metadata from OCR-processed text. The research found that while regex approaches offer simplicity and speed, NLP models significantly outperform them in handling unstructured, real-world data.

## Key Findings
- **Regex-based approach**: Achieved 100% recall but only 11.41% precision (F1 score: 20.41%)
- **BERT conditional model**: Achieved 100% recall and 57% precision (F1 score: 72.61%)
- **T5 model**: Achieved 100% recall and 57.39% precision (F1 score: 72.93%)

## Dataset
The dataset used contains 200,000 labeled product images with various entity types:
- Item weight
- Item volume
- Voltage
- Wattage
- Maximum weight recommendation
- Height
- Width
- Depth

Each sample consists of an image link, image ID, entity name, and corresponding entity value.

## Pipeline Components

### 1. Data Preprocessing
- Image collection and labeling
- OCR text extraction using EasyOCR
- Text cleaning and normalization

### 2. Entity Extraction Methods
- **Regex-based approach**: Manual pattern creation for each entity type
- **BERT model**: Implementation using BartForConditionalGeneration with facebook/bart-base pre-trained weights
- **T5 model**: Sequence-to-sequence approach with the T5-base model

### 3. Evaluation Metrics
- Precision
- Recall
- F1 Score
- Processing speed
- Error analysis

## Requirements
```
python>=3.8
pytorch>=1.9.0
transformers>=4.10.0
easyocr>=1.4.1
numpy>=1.19.5
pandas>=1.3.0
scikit-learn>=0.24.2
```

## Installation
```bash
git clone https://github.com/username/entity-extraction.git
cd entity-extraction
pip install -r requirements.txt
```

## Usage

### Data Preparation
```python
# Load and preprocess images
python preprocess.py --input_dir /path/to/images --output_dir /path/to/output
```

### Training Models
```python
# Train BERT model
python train.py --model bert --data_path /path/to/data --epochs 10 --batch_size 16

# Train T5 model
python train.py --model t5 --data_path /path/to/data --epochs 7 --batch_size 8
```

### Evaluation
```python
# Evaluate models
python evaluate.py --model_path /path/to/model --test_data /path/to/test_data
```

### Inference
```python
# Extract entities from new images
python extract.py --image_path /path/to/image --model_path /path/to/model
```

## Future Work
- Improve model performance by creating hybrid approaches combining rule-based and NLP methods
- Optimize hyperparameters for better T5 performance
- Explore models that can better recognize text in different orientations and angles
- Address computational efficiency concerns for real-time applications

## Citation
```
@article{kamra2025comparative,
  title={Comparative Analysis of NLP and Rule-Based Approaches for Entity Extraction from Product Images Using OCR},
  author={Kamra, Shubh and Dahiya, Himanshu and Mahato, Shubhomakar and Yadav, Jatin and Agarwal, Mohit},
  journal={},
  year={2025}
}
```

## Contributor
- Shubh Kamra
- Himanshu Dahiya
- Shubhomakar Mahato
- Jatin Yadav
- Mohit Agarwal
  

