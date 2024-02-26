# Suicide Risk Detection in Social Media Posts

## Project Overview
This project aims to use advanced machine learning techniques, specifically Recurrent Neural Networks (RNNs) and Transformers, to detect potential suicide risk from social media posts. The goal is to categorize posts into 'Suicide' and 'Non-Suicide' to facilitate timely intervention for individuals in distress.

## Dataset
The data is from kaggle suicide detection dataset. The dataset includes anonymized social media posts labeled for suicide risk. https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

## Model Architecture
The model employs a two-step approach:
1. **RNN Layer:** For capturing temporal dependencies within texts.
2. **Transformer Encoder:** For applying self-attention mechanisms, enhancing contextual understanding.

## Preprocessing
Preprocessing steps include cleaning, tokenization, and vectorization, preparing raw text for the model.

## Requirements
- TensorFlow
- Keras
- NLTK
- NumPy

Refer to `requirements.txt` for a full list.

## Quick Start
1. Clone the repository: git clone [https://github.com/your-repository.git](https://github.com/Mehrads/NLP-method)https://github.com/Mehrads/NLP-method
2. Install dependencies: pip install -r requirements.txt
3. Run the model: python quick-start.py





