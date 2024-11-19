# Import libraries

import pandas as pd
import numpy as np
import tensorflow as tf

# Load the IMDB word index
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the prebuilt SimpleRNN model
model = tf.keras.models.load_model('simple_rnn_imdb.h5')

# Function to decode the review
def decode_review(review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in review])

# Function to encode the review
from tensorflow.keras.preprocessing import sequence 
def encode_review(review):
    review = review.lower().split()
    encoded_review = [word_index.get(i, 2)+3 for i in review]
    encoded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return encoded_review

# Function to predict the sentiment
def predict_sentiment(input):
    prediction = model.predict(input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    sentiment = 'Neutral' if prediction[0][0] == 0.5 else sentiment
    return sentiment, prediction


# Streamlit app
import streamlit as st

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

# User input
review = st.text_area('Give your review here')
if st.button('Classify'):
    # encoding the review
    encoded_review = encode_review(review)

    # predict the sentiment
    sentiment, score = predict_sentiment(encoded_review)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score.item():.2f}')
else:
    st.write('Please enter a movie review')
