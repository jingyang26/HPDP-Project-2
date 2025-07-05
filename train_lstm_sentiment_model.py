# train_lstm_sentiment_model.py

import os
import re
import json
import pickle
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

# === SUPPRESS ALL WARNINGS ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # Hide TensorFlow INFO & WARN
warnings.filterwarnings("ignore")            # Hide Python warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Hide TensorFlow logs

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# === Setup ===
Path("logs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/lstm_training.log")
    ]
)
logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class LSTMSentimentTrainer:
    def __init__(self, dataset_path: str = "dataset.csv"):
        self.dataset_path = dataset_path
        self.stopwords = set(stopwords.words('english')).union({
            'malaysia', 'malaysian', 'kuala', 'lumpur', 'kl', 'penang', 'langkawi',
            'go', 'going', 'went', 'visit', 'visiting', 'visited', 'trip', 'travel',
            'place', 'places', 'time', 'day', 'days', 'week', 'month', 'year',
            'like', 'would', 'could', 'should', 'really', 'also', 'get', 'got'
        })
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_words = 10000
        self.max_length = 100

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+", '', text)
        text = re.sub(r'/u/\w+|/r/\w+|\[deleted\]|\[removed\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        return 'n'

    def preprocess_text(self, text: str) -> str:
        text = self.clean_text(text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha() and t not in self.stopwords]
        tagged = pos_tag(tokens)
        lemmatized = [self.lemmatizer.lemmatize(w, self.get_wordnet_pos(p)) for w, p in tagged]
        return ' '.join(lemmatized)

    def load_and_prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Loading dataset from: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df[df['language'] == 'en'].copy()

        # Label mapping (Option A)
        label_map = {
            'positive': 'positive', 'pos': 'positive', '1': 'positive',
            'negative': 'negative', 'neg': 'negative', '-1': 'negative',
            'neutral': 'neutral', 'uncertainty': 'neutral', 'uncertain': 'neutral', '0': 'neutral'
        }
        df['label'] = df['label'].astype(str).str.lower().str.strip()
        df['label'] = df['label'].map(label_map)
        df = df[df['label'].notna() & df['text'].notna()]

        logger.info(f"Preprocessing {len(df)} texts...")
        df['cleaned'] = df['text'].apply(self.preprocess_text)

        logger.info("Tokenizing...")
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(df['cleaned'])
        sequences = self.tokenizer.texts_to_sequences(df['cleaned'])
        padded = pad_sequences(sequences, maxlen=self.max_length)

        logger.info("Encoding labels...")
        y = self.label_encoder.fit_transform(df['label'])
        y_cat = to_categorical(y)

        return padded, y_cat

    def build_model(self, output_dim: int):
        logger.info("Building LSTM model...")
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_length))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        logger.info("Starting LSTM training pipeline...")
        X, y = self.load_and_prepare_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        model = self.build_model(output_dim=y.shape[1])
        history = model.fit(X_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

        logger.info("Evaluating model...")
        loss, acc = model.evaluate(X_test, y_test)
        logger.info(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred).tolist()

        # Save everything
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/lstm_model_{timestamp}.h5"
        tokenizer_path = f"models/tokenizer_{timestamp}.pkl"
        label_encoder_path = f"models/label_encoder_{timestamp}.pkl"
        report_path = f"reports/lstm_training_report_{timestamp}.json"

        model.save(model_path)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open(label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder, f)

        report_data = {
            "timestamp": timestamp,
            "model_path": model_path,
            "label_encoder": label_encoder_path,
            "tokenizer": tokenizer_path,
            "labels": list(self.label_encoder.classes_),
            "test_accuracy": acc,
            "confusion_matrix": matrix,
            "classification_report": report,
            "strategy": "Option A (uncertainty → neutral)",
            "language_filter": "en"
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved: {model_path}")
        logger.info(f"Report saved: {report_path}")
        logger.info("LSTM training completed successfully!")

def main():
    trainer = LSTMSentimentTrainer("dataset.csv")
    trainer.train()

if __name__ == "__main__":
    main()
