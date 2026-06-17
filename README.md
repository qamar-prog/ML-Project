This repository contains a Google Colab notebook demonstrating sentiment analysis on a Twitter dataset. The project involves data loading, preprocessing, training and evaluating two different machine learning models (LSTM and BERT), and comparing their performance.

Table of Contents
Project Overview
Dataset
Setup and Installation
Data Loading and Initial Exploration
Data Preprocessing
LSTM Model
BERT Model
Model Comparison
How to Run
Project Overview
The goal of this project is to classify the sentiment of tweets as either positive (1) or negative (0). We explore two popular deep learning approaches:

Long Short-Term Memory (LSTM) Network: A type of recurrent neural network well-suited for sequence data like text.
BERT (Bidirectional Encoder Representations from Transformers): A state-of-the-art transformer-based model known for its strong performance in various NLP tasks.
Dataset
The dataset used is the "Sentiment140" dataset, which contains 1.6 million tweets classified as positive (4) or negative (0). The notebook preprocesses these labels to 0 and 1 respectively.

Source: training.1600000.processed.noemoticon.csv

Setup and Installation
To run this notebook, you'll need a Google Colab environment. The necessary libraries are installed within the notebook itself.

# Install Hugging Face Transformers and Datasets
!pip install transformers datasets sentencepiece --quiet
Data Loading and Initial Exploration
The data is loaded from Google Drive into a pandas DataFrame. Initial steps involve inspecting the data's structure and selecting relevant columns.

import pandas as pd

col_names = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(
    "/content/drive/MyDrive/Colab Notebooks/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=col_names
)

# Keep only relevant columns and map sentiment values
data = data[['sentiment', 'text']]
data['sentiment'] = data['sentiment'].map({0:0, 4:1})
Data Preprocessing
Text data undergoes cleaning to remove URLs, mentions, hashtags, punctuation, and extra spaces. The data is then split into training and testing sets.

def clean_text(text):
    # Lowercase, remove URLs, mentions, hashtags, punctuation, numbers, and extra spaces
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['clean_text'] = data['text'].apply(clean_text)

X = data['clean_text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
LSTM Model
Tokenization and Padding
Text data is tokenized and padded to a uniform length.

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')
Model Architecture
A Bidirectional LSTM model with an embedding layer and dropout is used for sentiment classification.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Training and Evaluation
The model is trained and evaluated, and performance metrics like accuracy, precision, recall, and F1-score are reported.

EPOCHS = 5
BATCH_SIZE = 128

history = model.fit(
    X_train_pad, y_train_enc,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluation
loss, accuracy = model.evaluate(X_test_pad, y_test_enc)
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

from sklearn.metrics import confusion_matrix, classification_report

print("Test Accuracy:", accuracy)
print(classification_report(y_test_enc, y_pred))
BERT Model
Dataset Preparation for Hugging Face
The dataset is converted into a Hugging Face Dataset object and a smaller subset is used for faster training.

from datasets import Dataset

train_dict = {"text": X_train.tolist(), "label": y_train.tolist()}
test_dict = {"text": X_test.tolist(), "label": y_test.tolist()}

train_dataset = Dataset.from_dict(train_dict).shuffle(seed=42).select(range(200000))
test_dataset = Dataset.from_dict(test_dict).shuffle(seed=42).select(range(20000))
Tokenization
A BERT tokenizer is used to tokenize the text, ensuring proper formatting for the BERT model.

from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
Model Loading and Training
A pretrained bert-base-uncased model is loaded and fine-tuned on the sentiment analysis task using the Hugging Face Trainer API.

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./bert-twitter-sentiment",
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
Evaluation
The BERT model's performance is evaluated, and a classification report and confusion matrix are generated.

results = trainer.evaluate()
print("Evaluation results:", results)

preds = trainer.predict(test_dataset)
y_pred_bert = preds.predictions.argmax(-1)
y_true_bert = preds.label_ids

print(classification_report(y_true_bert, y_pred_bert))
Model Comparison
The performance of the LSTM and BERT models are compared using accuracy, precision, recall, and F1-score, visualized using bar charts.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Assuming `cr_lstm` and `cr_bert` are the classification reports for LSTM and BERT respectively
# Example structure for cr_lstm and cr_bert (you would get these from your classification_report output):
# cr_lstm = {'0': {'precision': ..., 'recall': ..., 'f1-score': ...}, '1': {...}, 'accuracy': ...}
# cr_bert = {'0': {'precision': ..., 'recall': ..., 'f1-score': ...}, '1': {...}, 'accuracy': ...}

# Placeholder for demonstration, replace with actual results from your previous cells
cr_lstm = {
    '0': {'precision': 0.82, 'recall': 0.82, 'f1-score': 0.82},
    '1': {'precision': 0.83, 'recall': 0.83, 'f1-score': 0.83},
    'accuracy': 0.825
}

cr_bert = {
    '0': {'precision': results['eval_precision'], 'recall': results['eval_recall'], 'f1-score': results['eval_f1']},
    '1': {'precision': results['eval_precision'], 'recall': results['eval_recall'], 'f1-score': results['eval_f1']},
    'accuracy': results['eval_accuracy']
}

metrics = ['precision', 'recall', 'f1-score', 'accuracy']

lstm_scores = [cr_lstm['1'][m] if m != 'accuracy' else cr_lstm['accuracy'] for m in metrics]
bert_scores = [cr_bert['1'][m] if m != 'accuracy' else cr_bert['accuracy'] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, lstm_scores, width, label='LSTM')
ax.bar(x + width/2, bert_scores, width, label='BERT')

ax.set_ylabel("Score")
ax.set_title("Model Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0,1)
ax.legend()
plt.show()
How to Run
Open in Google Colab: Upload the .ipynb file to Google Drive and open it with Google Colab.
Mount Google Drive: Execute the cell to mount your Google Drive to access the dataset.
Run All Cells: Go to Runtime -> Run all to execute the entire notebook. Ensure you have the dataset (training.1600000.processed.noemoticon.csv) in the specified path within your Google Drive (/content/drive/MyDrive/Colab Notebooks/).
WandB Login (for BERT): During BERT model training, you might be prompted to log in to Weights & Biases. You can either log in or choose to skip it if you don't need experiment tracking.
