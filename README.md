# ML_CBP_FL_sentimental_Analysis

# Federated Learning for Sentiment Analysis on IMDB Movie Reviews

This project implements a complete **Federated Learning (FL) pipeline** for binary sentiment classification using the **Flower (flwr)** framework and **Scikit-Learn**. It demonstrates how to train a machine learning model on decentralized data while maintaining privacy.

## Project Overview

The core objective is to perform sentiment analysis (Positive/Negative) on the **IMDB Movie Reviews** dataset using a Federated Learning approach.

### Key Components:
| Component | Detail |
|---|---|
| **Dataset** | IMDB Movie Reviews (50,000 reviews) |
| **Task** | Binary Sentiment Classification |
| **Feature Engineering** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **FL Framework** | [Flower (flwr)](https://flower.dev/) |
| **Aggregation Strategy**| FedAvg (Federated Averaging) |
| **Data Partitioning** | IID and Non-IID distributions |
| **Model** | Logistic Regression (SGDClassifier) |
| **Environment** | Jupyter Notebook / Google Colab |

## 🛠️ Installation

To run this notebook, you need to install the following dependencies:

```bash
pip install flwr==1.8.0 scikit-learn pandas numpy matplotlib seaborn
```

## 📂 Project Structure

- `fl_sentiment_imdb.ipynb`: The main notebook containing the data preprocessing, FL simulation, and evaluation.
- `IMDB Dataset.csv`: Local dataset file used for analysis.
- `https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`: Dataset source.

## Workflow

1.  **Data Preprocessing**: Cleaning text (removing HTML tags, special characters, lowercasing) and label encoding.
2.  **Feature Extraction**: Converting text into numerical vectors using `TfidfVectorizer`.
3.  **Data Partitioning**:
    *   **IID**: Data is shuffled and distributed evenly across clients.
    *   **Non-IID**: Data is sorted by label before distribution, creating a "class-imbalance" scenario for each client (simulating real-world drift).
4.  **Federated Simulation**:
    *   Initialize a central Server with a `FedAvg` strategy.
    *   Launch multiple Clients.
    *   Perform rounds of local training and global aggregation.
5.  **Evaluation**: Compare the Federated model performance against a traditional Centralized model using Accuracy, Precision, Recall, and Confusion Matrices.

##  Results Summary

The project provides insights into:
- How Federated Learning performs compared to Centralized Learning.
- The impact of Non-IID data on convergence and final accuracy.
- Visualization of sentiment distribution and review lengths.

