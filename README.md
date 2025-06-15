
# Name-Enity

🧠 Named Entity Recognition using BiLSTM
A deep learning project focused on Named Entity Recognition (NER) using a Bidirectional LSTM (BiLSTM) architecture with TensorFlow and Keras. This project trains a model to identify entities such as persons, organizations, and locations within sentences using annotated data.

📚 Table of Contents
Introduction

Features

Installation

Usage

Model Architecture

Examples

Dependencies

Configuration

Troubleshooting

Contributors

License

📝 Introduction
Named Entity Recognition (NER) is a key task in Natural Language Processing (NLP) that involves locating and classifying named entities in text. This project implements a deep learning approach to NER using a BiLSTM-based model, trained on a labeled dataset. The solution includes data preprocessing, model training, evaluation, and real-time entity prediction.

✨ Features
Preprocessing and cleaning of NER-labeled datasets

Conversion of words and tags into indexed formats

Sequence padding and one-hot encoding

Construction of a deep learning model using BiLSTM layers

Model evaluation on validation data

Custom entity prediction for user-defined sentences

Integration with spaCy for comparison

⚙️ Installation
This project is designed to run seamlessly in Google Colab, but can also be executed locally with Python 3.7+ and the necessary packages installed. All dependencies are listed in the dependencies section below.

🚀 Usage
Upload the labeled NER dataset (ner_dataset.csv)

Preprocess the data and split into training and validation sets

Train the BiLSTM model

Evaluate performance on unseen data

Input your own sentences to extract named entities

Optionally, compare predictions with spaCy's built-in NER model

🧠 Model Architecture
The model is composed of:

An embedding layer for dense vector representation of words

A Bidirectional LSTM to capture context from both directions

A second LSTM layer to improve depth and context understanding

A TimeDistributed dense layer for sequence-level tagging

A softmax activation for predicting entity classes per token

🧪 Examples
The trained model can identify named entities in new sentences and output labeled tokens, such as detecting names of people, organizations, and geographic locations. A side-by-side comparison with spaCy’s predictions can also be performed to benchmark performance.

📦 Dependencies
TensorFlow

Keras

NumPy

Pandas

Scikit-learn

spaCy

The model also uses the en_core_web_sm model from spaCy for optional benchmarking.

⚙️ Configuration
The maximum sequence length is dynamically set based on the longest sentence in the dataset

A batch size of 32 is used during training

Training is conducted over 5 epochs

10% of the dataset is reserved for validation

All configurations can be customized within the notebook as needed.

🛠 Troubleshooting
Ensure the uploaded dataset has no malformed rows or missing tags

For memory constraints, it is recommended to use Google Colab

If predictions seem inaccurate, consider increasing training epochs or reviewing data preprocessing steps


