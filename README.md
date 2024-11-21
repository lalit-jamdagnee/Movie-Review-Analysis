
# Movie Review Sentiment Analysis

This project applies **sentiment analysis** to IMDB movie reviews using a **Recurrent Neural Network (RNN)** with a **Keras Embedding layer**. The model classifies reviews as either **positive** or **negative**, demonstrating basic NLP and machine learning techniques.

## Features
- Sentiment classification of movie reviews.
- Simple yet effective architecture using **Simple RNN**.
- Custom embedding layer for word representations.
- Python implementation with **Keras** and **TensorFlow** frameworks.
- Pre-trained model weights included for quick evaluation.

## Dataset
The project uses the **IMDB movie review dataset**, which consists of 50,000 labeled reviews:
- **Training set**: 25,000 reviews.
- **Test set**: 25,000 reviews.

Each review is preprocessed by:
1. Tokenizing the text into sequences of integers.
2. Padding sequences to a fixed length.
3. Converting sequences into embeddings using the `keras.layers.Embedding` layer.

## Model Architecture
The model follows a simple structure:
- **Embedding Layer**: Maps words to dense vectors.
- **Simple RNN Layer**: Captures sequential dependencies in the text.
- **Dense Layer**: Fully connected layer for classification.
- **Output Layer**: Sigmoid activation for binary output (positive or negative sentiment).

### Model Summary
| Layer           | Output Shape | Description                      |
|------------------|--------------|----------------------------------|
| Embedding        | (None, 100, 128) | Converts words into dense vectors. |
| SimpleRNN        | (None, 128)  | Processes sequential text.      |
| Dense            | (None, 1)    | Final binary classification.    |

## Performance
The model achieves approximately **85% accuracy** on the test set, balancing simplicity with performance.

## Technologies Used
- **Python**: Core programming language.
- **Keras**: Model building and training.
- **TensorFlow**: Backend for deep learning computations.
- **NumPy**: Numerical operations.
- **Pandas**: Data manipulation.

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/lalit-jamdagnee/Movie-Review-Analysis.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Test the model with a custom review:
   ```bash
   python predict.py -r "Enter your movie review here."
   ```

## Live Demo
The project features a **Streamlit app** for an interactive experience. You can test the model with your own reviews directly on the live platform! 

Here is the link for the Implementation.(check out for live demo): https://movie-review-analysis-at8vw4eovjgbxrynonmbrg.streamlit.app/

## Project Files
- `train.py`: Script for training the RNN model.
- `predict.py`: Script for predicting sentiment on custom reviews.
- `model.h5`: Saved weights of the trained model.
- `requirements.txt`: List of dependencies.
- `README.md`: Documentation for the project.

## Future Enhancements
- Use more advanced models like LSTM or GRU for better context capture.
- Add real-time review scraping and analysis.
- Explore multi-class sentiment labels (e.g., "positive," "negative," "neutral").
- Integrate attention mechanisms for interpretability.

## References
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Keras Documentation](https://keras.io/)
- [Streamlit Framework](https://streamlit.io/)


