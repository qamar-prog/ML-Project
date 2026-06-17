# Twitter Sentiment Analysis using LSTM and BERT

## Overview

This project performs sentiment analysis on Twitter data using two deep learning approaches:

* **Bidirectional LSTM (Long Short-Term Memory)**
* **BERT (Bidirectional Encoder Representations from Transformers)**

The objective is to classify tweets as **Positive** or **Negative** and compare the performance of traditional deep learning and transformer-based architectures.

---

## Dataset

The project uses the **Sentiment140 Dataset**, containing 1.6 million labeled tweets.

* Positive Sentiment → 1
* Negative Sentiment → 0

Dataset Features:

* Tweet Text
* Sentiment Label

---

## Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* TensorFlow / Keras
* Hugging Face Transformers
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Datasets

---

## Data Preprocessing

The following preprocessing steps were applied:

* Convert text to lowercase
* Remove URLs
* Remove mentions (@user)
* Remove hashtags (#)
* Remove punctuation and special characters
* Remove extra spaces

Example:

Original:

```text
@user I love this product! https://example.com #awesome
```

Processed:

```text
i love this product awesome
```

---

## LSTM Model

### Architecture

* Embedding Layer
* Bidirectional LSTM (128 Units)
* Dropout Layer (0.5)
* Dense Output Layer (Sigmoid)

### Training Configuration

* Vocabulary Size: 10,000
* Maximum Sequence Length: 100
* Embedding Dimension: 100
* Batch Size: 128
* Epochs: 5

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## BERT Model

### Model

* bert-base-uncased

### Training Configuration

* Learning Rate: 2e-5
* Batch Size: 32
* Epochs: 3
* Weight Decay: 0.01
* Maximum Length: 64

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report

---

## Model Comparison

The performance of both models was compared using:

* Accuracy
* Precision
* Recall
* F1-Score

A visualization was created using Matplotlib to compare model performance.

| Metric    | LSTM                 | BERT             |
| --------- | -------------------- | ---------------- |
| Accuracy  | Better than baseline | State-of-the-art |
| Precision | Evaluated            | Evaluated        |
| Recall    | Evaluated            | Evaluated        |
| F1-Score  | Evaluated            | Evaluated        |

> Replace the table values with your actual experimental results.

---

## Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── Twitter_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
└── dataset/
```

---

## How to Run

### Clone Repository

```bash
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

Open the notebook in:

* Google Colab
* Jupyter Notebook

Upload the Sentiment140 dataset and execute all cells.

---

## Key Learning Outcomes

* Natural Language Processing (NLP)
* Deep Learning for Text Classification
* Transformer-based Models (BERT)
* Sentiment Analysis
* Data Preprocessing
* Model Evaluation and Comparison

---

## Future Improvements

* RoBERTa Fine-Tuning
* DistilBERT Optimization
* Hyperparameter Tuning
* Deployment using Streamlit or Gradio
* Real-Time Twitter Sentiment Monitoring

---

## Author

**Qamar Abbas**

BS Artificial Intelligence
FAST – National University of Computer and Emerging Sciences (FAST-NUCES)

Skills:
Machine Learning • Deep Learning • NLP • Explainable AI • Python
