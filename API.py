# Import Library
import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
from flask import Flask, request, jsonify
from flasgger import Swagger
from datetime import datetime


# Function untuk Preprocessing Text
def remove_newline(text):
    return text.replace('\n', ' ')

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_double_spaces(text):
    return re.sub(r'\s+', ' ', text)

def remove_stopwords_english(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def remove_stopwords_indonesian(text):
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def tokenize(text):
    return word_tokenize(text)

def stem_indonesian(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def stem_english(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def convert_tokens_to_sentence(tokens):
    return ' '.join(tokens)

def text_processing_pipeline(text):
    text = remove_newline(text)
    text = remove_special_characters(text)
    text = remove_double_spaces(text)
    text = remove_stopwords_english(text)
    text = remove_stopwords_indonesian(text)
    tokens = tokenize(text)
    tokens = [stem_indonesian(token) for token in tokens]
    tokens = [stem_english(token) for token in tokens]
    text = convert_tokens_to_sentence(tokens)
    return text

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)

# Load Model dan Vectorizer Sebelumnya
loaded_model = joblib.load('logreg.joblib')
loaded_vectorizer = joblib.load('TF-IDF.joblib')

# Define API endpoints
# Endpoints Single Predict
@app.route('/label', methods=['POST'])
def single_predict():
    """
    ENDPOINT FOR TEXT CLASSIFICATION USING SINGLE PREDICT
    ---
    parameters:
      - name: text
        in: body
        type: string
        required: true
    responses:
      200:
        description: Text to be predicted.
    """
    start_time = datetime.now()
    
    data = request.get_json()
    if isinstance(data, dict):
        text = data['text']
    else:
        return jsonify({'error': 'Invalid input format'}), 400
    
    processed_text = text_processing_pipeline(text)
    vectorized_text = loaded_vectorizer.transform([processed_text])
    prediction = loaded_model.predict(vectorized_text)[0]

    label_mapping = {0: 'Negative', 1: 'Positive'}
    mapped_label = label_mapping[prediction]
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    response_data = {
        "label": mapped_label,
        "text": text
    }

    response_json = {
        "data": response_data,
        "description": "Text Classification Predicted",
        "processing time": f"{processing_time:.9f} seconds",
        "status_code": 200
    }

    return jsonify(response_json), 200

# Endpoints Batch Predict
@app.route('/label/batch', methods=['POST'])
def batch_predict():
    """
    ENDPOINT FOR TEXT CLASSIFICATION USING BATCH PREDICT
    ---
    parameters:
      - name: text_list
        in: body
        type: list
        required: true
        items:
          type: object
          properties:
            text:
              type: string
    responses:
      200:
        description: Text to be predicted.
    """
    start_time = datetime.now()

    data = request.get_json()

    if isinstance(data, list):
        text_list = [item.get('text', '') for item in data]
    else:
        return jsonify({'error': 'Invalid input format'}), 400

    processed_texts = [text_processing_pipeline(text) for text in text_list]
    vectorized_texts = loaded_vectorizer.transform(processed_texts)
    predictions = loaded_model.predict(vectorized_texts).tolist()

    # Mapping nilai label yang dihasilkan model ke dalam bentuk yang diinginkan
    label_mapping = {0: 'Negative', 1: 'Positive'}
    results = [{'label': label_mapping[label], 'text': text} for text, label in zip(text_list, predictions)]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    response_json = {
        'data': results,
        'description': 'Text Classification Predicted',
        'processing time': f'{processing_time:.9f} seconds',
        'status_code': 200
    }

    return jsonify(response_json), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
