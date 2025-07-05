"""
Malaysian Tourism Naive Bayes Sentiment Model Trainer
====================================================

This script trains a Naive Bayes classifier on the tourism dataset with
text, language, and label columns. Filters for English text and maps
uncertainty to neutral.

Features:
- Language filtering (English only)
- Three-class sentiment: positive, negative, neutral (uncertainty→neutral)
- Naive Bayes model training with cross-validation
- Text preprocessing using NLTK (same as consumer)
- Model evaluation and performance metrics
- Model persistence for integration with consumer

Author: Big Data & NLP Analytics Team
Date: July 2, 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Import NLTK preprocessing from your consumer
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import pos_tag
    import re
    NLP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NLTK not available: {e}")
    NLP_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/naive_bayes_training.log')
    ]
)
logger = logging.getLogger(__name__)

class MalaysianTourismNBTrainer:
    """Naive Bayes model trainer for Malaysian tourism sentiment analysis"""
    
    def __init__(self, dataset_path: str = "dataset.csv"):
        """Initialize the trainer"""
        self.dataset_path = dataset_path
        self.setup_directories()
        self.setup_preprocessing()
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['models', 'logs', 'reports', 'data/processed']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("✅ Training directories created/verified")
    
    def setup_preprocessing(self):
        """Setup text preprocessing (same as consumer)"""
        if NLP_AVAILABLE:
            # Download NLTK resources
            resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for resource in resources:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
            
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            
            # Load stopwords (same as consumer)
            try:
                english_stopwords = set(stopwords.words('english'))
                custom_stopwords = {
                    'malaysia', 'malaysian', 'kuala', 'lumpur', 'kl', 'penang', 'langkawi',
                    'go', 'going', 'went', 'visit', 'visiting', 'visited', 'trip', 'travel',
                    'place', 'places', 'time', 'day', 'days', 'week', 'month', 'year',
                    'like', 'would', 'could', 'should', 'really', 'also', 'get', 'got'
                }
                self.stopwords = english_stopwords.union(custom_stopwords)
                logger.info(f"✅ Preprocessing setup with {len(self.stopwords)} stopwords")
            except:
                self.stopwords = set()
                logger.warning("⚠️ Could not load stopwords")
        else:
            self.stemmer = None
            self.lemmatizer = None
            self.stopwords = set()
    
    def load_dataset(self) -> pd.DataFrame:
        """Load and validate the three-column dataset (text, language, label)"""
        try:
            logger.info(f"📂 Loading dataset from {self.dataset_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(self.dataset_path, encoding=encoding)
                    logger.info(f"✅ Dataset loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not load dataset with any encoding")
            
            logger.info(f"📊 Dataset shape: {df.shape}")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            # Validate required columns
            required_columns = ['text', 'language', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Try to find similar column names
                available_cols = df.columns.tolist()
                logger.warning(f"⚠️ Missing required columns: {missing_columns}")
                logger.info(f"📋 Available columns: {available_cols}")
                
                # Auto-detect similar column names
                column_mapping = {}
                for req_col in required_columns:
                    for avail_col in available_cols:
                        if req_col.lower() in avail_col.lower() or avail_col.lower() in req_col.lower():
                            column_mapping[req_col] = avail_col
                            break
                
                if column_mapping:
                    logger.info(f"🔍 Auto-detected column mapping: {column_mapping}")
                    df = df.rename(columns={v: k for k, v in column_mapping.items()})
                else:
                    raise ValueError(f"Required columns not found: {missing_columns}")
            
            # Display basic info
            logger.info("📈 Dataset info:")
            logger.info(f"  Total rows: {len(df)}")
            logger.info(f"  Missing values: {df.isnull().sum().sum()}")
            
            # Show language distribution
            if 'language' in df.columns:
                lang_counts = df['language'].value_counts()
                logger.info("🌐 Language distribution:")
                for lang, count in lang_counts.head(10).items():
                    logger.info(f"  {lang}: {count} ({count/len(df)*100:.1f}%)")
            
            # Show label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info("🏷️ Original label distribution:")
                for label, count in label_counts.items():
                    logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def filter_and_map_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for English language and map labels according to Option A"""
        logger.info("🔧 Filtering and mapping dataset...")
        
        # Filter for English language only
        original_size = len(df)
        df_filtered = df[df['language'] == 'en'].copy()
        english_size = len(df_filtered)
        
        logger.info(f"🌐 Language filtering: {original_size} → {english_size} rows ({english_size/original_size*100:.1f}% English)")
        
        if english_size == 0:
            raise ValueError("No English language data found! Check 'language' column values.")
        
        # Clean and standardize labels
        df_filtered['label'] = df_filtered['label'].astype(str).str.lower().str.strip()
        
        # Show original label distribution after filtering
        original_labels = df_filtered['label'].value_counts()
        logger.info("🏷️ English data label distribution (before mapping):")
        for label, count in original_labels.items():
            logger.info(f"  {label}: {count} ({count/len(df_filtered)*100:.1f}%)")
        
        # Option A: Map uncertainty → neutral, keep positive/negative
        label_mapping = {
            'positive': 'positive',
            'negative': 'negative',
            'uncertainty': 'neutral',  # Key mapping: uncertainty becomes neutral
            # Additional common variations
            'pos': 'positive',
            'neg': 'negative',
            'uncertain': 'neutral',
            'neutral': 'neutral',
            # Handle any numeric labels
            '1': 'positive',
            '0': 'neutral', 
            '-1': 'negative'
        }
        
        # Apply label mapping
        df_filtered['original_label'] = df_filtered['label'].copy()  # Keep original for reference
        df_filtered['label'] = df_filtered['label'].map(label_mapping)
        
        # Remove rows with unmapped labels
        unmapped_mask = df_filtered['label'].isna()
        unmapped_labels = df_filtered[unmapped_mask]['original_label'].unique()
        
        if len(unmapped_labels) > 0:
            logger.warning(f"⚠️ Found unmapped labels (will be removed): {unmapped_labels}")
        
        df_filtered = df_filtered[~unmapped_mask].copy()
        
        # Show final label distribution after mapping
        final_labels = df_filtered['label'].value_counts()
        logger.info("🎯 Final label distribution (after Option A mapping):")
        for label, count in final_labels.items():
            logger.info(f"  {label}: {count} ({count/len(df_filtered)*100:.1f}%)")
        
        # Check for class balance issues
        min_class_size = final_labels.min()
        max_class_size = final_labels.max()
        imbalance_ratio = max_class_size / min_class_size
        
        if imbalance_ratio > 10:
            logger.warning(f"⚠️ High class imbalance detected (ratio: {imbalance_ratio:.1f})")
            logger.warning("   Consider data balancing techniques if model performance is poor")
        
        logger.info(f"✅ Final dataset: {len(df_filtered)} samples with 3 classes")
        
        return df_filtered
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text (same logic as consumer)"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Basic cleaning
        text = self.basic_text_cleaning(text)
        
        if not NLP_AVAILABLE:
            return text.lower()
        
        try:
            # Tokenization
            tokens = word_tokenize(text.lower())
            
            # Remove non-alphabetic tokens
            alpha_tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
            
            # Remove stopwords
            filtered_tokens = [token for token in alpha_tokens if token not in self.stopwords]
            
            # POS tagging and lemmatization
            if filtered_tokens and self.lemmatizer:
                pos_tagged = pos_tag(filtered_tokens)
                processed_tokens = []
                
                for word, pos in pos_tagged:
                    wordnet_pos = self.get_wordnet_pos(pos)
                    lemmatized = self.lemmatizer.lemmatize(word, wordnet_pos)
                    processed_tokens.append(lemmatized)
                
                return ' '.join(processed_tokens)
            else:
                return ' '.join(filtered_tokens)
                
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return text.lower()
    
    def basic_text_cleaning(self, text: str) -> str:
        """Basic text cleaning (same as consumer)"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'/u/\w+', '', text)
        text = re.sub(r'/r/\w+', '', text)
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert TreeBank POS tags to WordNet format"""
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n'
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Prepare and clean the filtered dataset"""
        logger.info("🧹 Preparing and cleaning data...")
        
        # Remove rows with missing text
        df_clean = df.dropna(subset=['text', 'label']).copy()
        logger.info(f"📝 After removing NaN: {len(df_clean)} rows")
        
        # Remove empty texts
        df_clean = df_clean[df_clean['text'].str.strip() != '']
        logger.info(f"📝 After removing empty text: {len(df_clean)} rows")
        
        # Preprocess texts
        logger.info("🔧 Preprocessing texts...")
        texts = []
        labels = df_clean['label'].tolist()
        
        for i, text in enumerate(df_clean['text']):
            if i % 1000 == 0:
                logger.info(f"  Processed {i}/{len(df_clean)} texts")
            
            processed_text = self.preprocess_text(text)
            texts.append(processed_text)
        
        # Remove empty processed texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"✅ Final preprocessed dataset: {len(texts)} samples")
        
        # Final class distribution
        from collections import Counter
        final_distribution = Counter(labels)
        logger.info("📊 Final class distribution:")
        for label, count in final_distribution.items():
            logger.info(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
        
        return texts, labels
    
    def train_models(self, texts: List[str], labels: List[str]) -> Dict:
        """Train multiple Naive Bayes models with different configurations"""
        logger.info("🚀 Starting model training...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        logger.info(f"🏷️ Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Split data with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"📊 Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Show train/test distribution
        from collections import Counter
        train_dist = Counter([self.label_encoder.classes_[y] for y in y_train])
        test_dist = Counter([self.label_encoder.classes_[y] for y in y_test])
        
        logger.info("📈 Training set distribution:")
        for label, count in train_dist.items():
            logger.info(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Model configurations optimized for 3-class sentiment
        configurations = {
            'multinomial_tfidf': {
                'vectorizer': TfidfVectorizer(
                    max_features=15000, 
                    ngram_range=(1, 2), 
                    min_df=2,
                    max_df=0.95
                ),
                'classifier': MultinomialNB(alpha=0.5)
            },
            'multinomial_count': {
                'vectorizer': CountVectorizer(
                    max_features=15000, 
                    ngram_range=(1, 2), 
                    min_df=2,
                    max_df=0.95
                ),
                'classifier': MultinomialNB(alpha=0.5)
            },
            'complement_tfidf': {
                'vectorizer': TfidfVectorizer(
                    max_features=15000, 
                    ngram_range=(1, 2), 
                    min_df=2,
                    max_df=0.95
                ),
                'classifier': ComplementNB(alpha=0.5)
            }
        }
        
        results = {}
        
        for name, config in configurations.items():
            logger.info(f"🔧 Training {name} model...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', config['vectorizer']),
                ('classifier', config['classifier'])
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            
            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
            
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(
                    y_test, y_pred, 
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        logger.info(f"🏆 Best model: {best_model_name}")
        
        return results, best_model_name, (X_test, y_test)
    
    def hyperparameter_tuning(self, texts: List[str], labels: List[str]) -> Dict:
        """Perform hyperparameter tuning for the best model"""
        logger.info("🎯 Starting hyperparameter tuning...")
        
        y_encoded = self.label_encoder.transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Parameter grid for MultinomialNB with TF-IDF
        param_grid = {
            'classifier__alpha': [0.1, 1.0, 2.0]  # Only tune alpha
        }
        
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model evaluation
        best_pipeline = grid_search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        tuned_results = {
            'pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }
        
        logger.info(f"🎯 Tuned model: Accuracy={accuracy:.3f}, F1={f1:.3f}")
        logger.info(f"🎯 Best parameters: {grid_search.best_params_}")
        
        return tuned_results
    
    def save_models(self, results: Dict, best_model_name: str, tuned_results: Dict):
        """Save trained models and metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best model
        best_model_path = f'models/naive_bayes_best_model_{timestamp}.pkl'
        with open(best_model_path, 'wb') as f:
            pickle.dump(results[best_model_name]['pipeline'], f)
        
        # Save tuned model
        tuned_model_path = f'models/naive_bayes_tuned_model_{timestamp}.pkl'
        with open(tuned_model_path, 'wb') as f:
            pickle.dump(tuned_results['pipeline'], f)
        
        # Save label encoder
        label_encoder_path = f'models/label_encoder_{timestamp}.pkl'
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save model metadata
        metadata = {
            'timestamp': timestamp,
            'best_model_name': best_model_name,
            'label_classes': self.label_encoder.classes_.tolist(),
            'models': {
                name: {
                    'accuracy': results[name]['accuracy'],
                    'precision': results[name]['precision'],
                    'recall': results[name]['recall'],
                    'f1_score': results[name]['f1_score'],
                    'cv_mean': results[name]['cv_mean'],
                    'cv_std': results[name]['cv_std']
                }
                for name in results.keys()
            },
            'tuned_model': {
                'best_params': tuned_results['best_params'],
                'accuracy': tuned_results['accuracy'],
                'f1_score': tuned_results['f1_score']
            },
            'file_paths': {
                'best_model': best_model_path,
                'tuned_model': tuned_model_path,
                'label_encoder': label_encoder_path
            }
        }
        
        metadata_path = f'models/model_metadata_{timestamp}.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Models saved:")
        logger.info(f"  Best model: {best_model_path}")
        logger.info(f"  Tuned model: {tuned_model_path}")
        logger.info(f"  Label encoder: {label_encoder_path}")
        logger.info(f"  Metadata: {metadata_path}")
        
        return metadata
    
    def generate_report(self, results: Dict, tuned_results: Dict, metadata: Dict):
        """Generate comprehensive training report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'reports/naive_bayes_training_report_{timestamp}.json'
        
        # Combine all results
        full_report = {
            'training_timestamp': timestamp,
            'dataset_info': {
                'source': self.dataset_path,
                'preprocessing_enabled': NLP_AVAILABLE
            },
            'model_comparison': results,
            'hyperparameter_tuning': tuned_results,
            'metadata': metadata,
            'recommendations': {
                'best_overall_model': 'tuned_model',
                'production_ready': True,
                'integration_notes': [
                    "Model can be integrated into reddit_tourism_consumer.py",
                    "Use the tuned model for best performance",
                    "Preprocessing pipeline matches consumer preprocessing"
                ]
            }
        }
        
        # Remove non-serializable objects
        for model_name in full_report['model_comparison']:
            del full_report['model_comparison'][model_name]['pipeline']
        del full_report['hyperparameter_tuning']['pipeline']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Training report saved: {report_path}")
        return report_path
    
    def train(self) -> Dict:
        """Main training pipeline with Option A implementation"""
        logger.info("🚀 Starting Naive Bayes model training pipeline")
        logger.info("🎯 Option A: uncertainty → neutral mapping")
        logger.info("=" * 60)
        
        try:
            # Load dataset
            df = self.load_dataset()
            
            # Filter for English and map labels (Option A)
            df_filtered = self.filter_and_map_data(df)
            
            # Prepare data
            texts, labels = self.prepare_data(df_filtered)
            
            if len(texts) < 100:
                raise ValueError(f"Insufficient data: only {len(texts)} samples")
            
            # Ensure we have all three classes
            unique_labels = set(labels)
            expected_labels = {'positive', 'negative', 'neutral'}
            
            if not expected_labels.issubset(unique_labels):
                missing = expected_labels - unique_labels
                logger.warning(f"⚠️ Missing expected labels: {missing}")
            
            # Train multiple models
            results, best_model_name, test_data = self.train_models(texts, labels)
            
            # Hyperparameter tuning
            tuned_results = self.hyperparameter_tuning(texts, labels)
            
            # Save models
            metadata = self.save_models(results, best_model_name, tuned_results)
            
            # Add Option A specific metadata
            metadata['training_strategy'] = 'option_a_uncertainty_to_neutral'
            metadata['label_mapping'] = {
                'positive': 'positive',
                'negative': 'negative', 
                'uncertainty': 'neutral'
            }
            metadata['language_filter'] = 'en'
            
            # Generate report
            report_path = self.generate_report(results, tuned_results, metadata)
            
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"🎯 Strategy: Option A (uncertainty → neutral)")
            logger.info(f"🌐 Language: English only")
            logger.info(f"🏷️ Classes: {self.label_encoder.classes_}")
            logger.info(f"🏆 Best model F1-score: {results[best_model_name]['f1_score']:.3f}")
            logger.info(f"🎯 Tuned model F1-score: {tuned_results['f1_score']:.3f}")
            logger.info(f"📊 Full report: {report_path}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        logger.info("🎯 Malaysian Tourism Sentiment Analysis - Option A Training")
        logger.info("📋 Strategy: English only, uncertainty → neutral")
        
        # Initialize trainer
        trainer = MalaysianTourismNBTrainer("dataset.csv")
        
        # Run training
        metadata = trainer.train()
        
        print("\n" + "="*60)
        print("🎉 NAIVE BAYES MODEL TRAINING COMPLETED!")
        print("🎯 OPTION A: uncertainty → neutral")
        print("="*60)
        print(f"📁 Models saved in: models/")
        print(f"📊 Reports saved in: reports/")
        print(f"🏷️ Classes: positive, negative, neutral")
        print(f"🔗 Integration ready for reddit_tourism_consumer.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())