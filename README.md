# IMDB Sentiment Analysis  

## Overview  
This project implements a sentiment analysis model to classify IMDB movie reviews as either positive or negative. Using deep learning techniques, the project demonstrates how to preprocess textual data and build an effective classification model with high accuracy.  

## Features  
- Sentiment classification of IMDB movie reviews with **90%+ accuracy**.  
- Preprocessing of over **50,000 reviews**, including tokenization and padding.  
- Implementation of an **LSTM-based deep learning model** for binary sentiment classification.  
- Real-time sentiment prediction for new reviews with **sub-second latency**.  

## Dataset  
The dataset used is the **IMDB Movie Reviews Dataset**, containing 50,000 labeled reviews (25,000 positive and 25,000 negative).  
- Download link: [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  

## Installation and Usage  
### Prerequisites  
- Python 3.7 or above  
- TensorFlow, Pandas, NumPy, Scikit-learn  

### Steps to Run  
1. Clone the repository:    
   git clone https://github.com/your-username/imdb-sentiment-analysis.git  
   cd imdb-sentiment-analysis  
2. Install required dependencies:
   pip install -r requirements.txt  
3. Place the IMDB Dataset (IMDB Dataset.csv) in the project directory.
4. Run the script to train the model:
   python train.py  
5. Test the model with a new review:
   python predict.py

### Model Architecture
1.Embedding Layer: Converts words into dense vector representations of fixed size (128 dimensions).
2.LSTM Layer: Captures sequential patterns with 128 units and dropout regularization.
3.Dense Layer: Outputs a single value (0 or 1) using a sigmoid activation function for binary classification.

### Results
Training Accuracy: 92%
Validation Accuracy: 90%
Test Accuracy: 89%
