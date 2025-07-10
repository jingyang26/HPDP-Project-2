#!/usr/bin/env python3
"""
Batch Pipeline for Malaysian Tourism Sentiment Analysis
======================================================

This pipeline orchestrates the complete batch processing workflow:
1. Data Collection from Reddit using data_collector.py
2. Train Naive Bayes sentiment model using train_naive_bayes_model.py
3. Train LSTM sentiment model using train_lstm_sentiment_model.py

Features:
- Sequential execution with dependency checking
- Comprehensive error handling and logging
- Progress tracking and timing
- Data validation between stages
- Configurable execution modes

Author: Big Data & NLP Analytics Team
Date: July 8, 2025
"""

import os
import sys
import time
import logging
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Ensure proper encoding for Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Setup logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/batch_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class MalaysianTourismBatchPipeline:
    """Complete batch processing pipeline for Malaysian tourism sentiment analysis"""
    
    def __init__(self, skip_collection: bool = False, skip_naive_bayes: bool = False, skip_lstm: bool = False):
        """Initialize the batch pipeline"""
        self.skip_collection = skip_collection
        self.skip_naive_bayes = skip_naive_bayes
        self.skip_lstm = skip_lstm
        
        self.pipeline_start_time = datetime.now()
        self.stage_times = {}
        self.stage_results = {}
        
        # File paths
        self.data_file = 'data/raw/malaysia_tourism_data.csv'
        self.nb_model_dir = 'models/naive_bayes'
        self.lstm_model_dir = 'models/lstm'
        
        # Minimum data requirements
        self.min_data_samples = 100
        self.min_samples_per_class = 10
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for the pipeline"""
        dirs = [
            'logs', 'data/raw', 'data/processed', 'data/analytics',
            'models', 'models/naive_bayes', 'models/lstm', 'reports'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info("SETUP: Pipeline directories created/verified")
    
    def log_stage_start(self, stage_name: str):
        """Log the start of a pipeline stage"""
        logger.info("=" * 60)
        logger.info(f"STAGE START: {stage_name}")
        logger.info("=" * 60)
        self.stage_times[stage_name] = {'start': datetime.now()}
        
    def log_stage_end(self, stage_name: str, success: bool = True, details: str = ""):
        """Log the end of a pipeline stage"""
        end_time = datetime.now()
        start_time = self.stage_times[stage_name]['start']
        duration = end_time - start_time
        
        self.stage_times[stage_name]['end'] = end_time
        self.stage_times[stage_name]['duration'] = duration
        self.stage_times[stage_name]['success'] = success
        
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"STAGE END: {stage_name} - {status}")
        logger.info(f"Duration: {duration}")
        if details:
            logger.info(f"Details: {details}")
        logger.info("=" * 60)
    
    def check_data_quality(self, file_path: str) -> Tuple[bool, Dict]:
        """Check if collected data meets quality requirements"""
        try:
            if not os.path.exists(file_path):
                return False, {"error": "Data file does not exist"}
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Basic statistics
            total_samples = len(df)
            
            # Check required columns
            required_columns = ['content', 'sentiment_label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, {"error": f"Missing required columns: {missing_columns}"}
            
            # Check data quality
            non_empty_content = df[df['content'].notna() & (df['content'].astype(str).str.strip() != '')].shape[0]
            valid_sentiments = df[df['sentiment_label'].notna()].shape[0]
            
            # Sentiment distribution
            sentiment_counts = df['sentiment_label'].value_counts().to_dict()
            min_class_samples = min(sentiment_counts.values()) if sentiment_counts else 0
            
            # Malaysia-related content
            malaysia_related = df[df.get('is_malaysia_related', False) == True].shape[0] if 'is_malaysia_related' in df.columns else 0
            
            quality_stats = {
                'total_samples': total_samples,
                'non_empty_content': non_empty_content,
                'valid_sentiments': valid_sentiments,
                'sentiment_distribution': sentiment_counts,
                'min_class_samples': min_class_samples,
                'malaysia_related': malaysia_related,
                'malaysia_percentage': (malaysia_related / total_samples * 100) if total_samples > 0 else 0
            }
            
            # Quality checks
            quality_issues = []
            
            if total_samples < self.min_data_samples:
                quality_issues.append(f"Insufficient total samples: {total_samples} < {self.min_data_samples}")
            
            if min_class_samples < self.min_samples_per_class:
                quality_issues.append(f"Insufficient samples per class: {min_class_samples} < {self.min_samples_per_class}")
            
            if non_empty_content < total_samples * 0.8:
                quality_issues.append(f"Too many empty content fields: {non_empty_content}/{total_samples}")
            
            if len(sentiment_counts) < 3:
                quality_issues.append(f"Missing sentiment classes: only {list(sentiment_counts.keys())}")
            
            is_quality_ok = len(quality_issues) == 0
            
            if quality_issues:
                quality_stats['issues'] = quality_issues
            
            return is_quality_ok, quality_stats
            
        except Exception as e:
            return False, {"error": f"Data quality check failed: {e}"}
    
    def run_data_collection(self) -> bool:
        """Run the data collection stage"""
        self.log_stage_start("Data Collection")
        
        try:
            # Check if data already exists and is recent
            if os.path.exists(self.data_file):
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.data_file))
                
                if file_age < timedelta(hours=24):  # Data is less than 24 hours old
                    logger.info(f"SKIP: Recent data file exists (age: {file_age})")
                    logger.info("Use --force-collection to override")
                    
                    # Still check data quality
                    is_quality_ok, quality_stats = self.check_data_quality(self.data_file)
                    
                    if is_quality_ok:
                        self.log_stage_end("Data Collection", True, f"Using existing data: {quality_stats['total_samples']} samples")
                        self.stage_results['data_collection'] = quality_stats
                        return True
                    else:
                        logger.warning("QUALITY: Existing data has quality issues, collecting new data")
                        logger.warning(f"Issues: {quality_stats.get('issues', quality_stats.get('error'))}")
            
            # Run data collection
            logger.info("EXECUTING: Running data_collector.py...")
            
            result = subprocess.run(
                [sys.executable, 'data_collector.py'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("SUCCESS: Data collection completed")
                logger.info("STDOUT:", result.stdout[-1000:] if result.stdout else "No output")
                
                # Validate collected data
                is_quality_ok, quality_stats = self.check_data_quality(self.data_file)
                
                if is_quality_ok:
                    self.log_stage_end("Data Collection", True, f"Collected {quality_stats['total_samples']} samples")
                    self.stage_results['data_collection'] = quality_stats
                    
                    # Log data statistics
                    logger.info("DATA STATS:")
                    logger.info(f"  Total samples: {quality_stats['total_samples']}")
                    logger.info(f"  Malaysia-related: {quality_stats['malaysia_related']} ({quality_stats['malaysia_percentage']:.1f}%)")
                    logger.info(f"  Sentiment distribution: {quality_stats['sentiment_distribution']}")
                    
                    return True
                else:
                    error_msg = f"Data quality check failed: {quality_stats.get('issues', quality_stats.get('error'))}"
                    self.log_stage_end("Data Collection", False, error_msg)
                    return False
            else:
                error_msg = f"Data collection failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr[-1000:]}"
                
                self.log_stage_end("Data Collection", False, error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "Data collection timed out after 1 hour"
            self.log_stage_end("Data Collection", False, error_msg)
            return False
            
        except Exception as e:
            error_msg = f"Data collection error: {e}"
            self.log_stage_end("Data Collection", False, error_msg)
            return False
    
    def run_naive_bayes_training(self) -> bool:
        """Run the Naive Bayes model training stage"""
        self.log_stage_start("Naive Bayes Training")
        
        try:
            # Check if model already exists
            model_files = list(Path(self.nb_model_dir).glob('*.pkl'))
            
            if model_files:
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                model_age = datetime.now() - datetime.fromtimestamp(latest_model.stat().st_mtime)
                
                if model_age < timedelta(hours=12):  # Model is less than 12 hours old
                    logger.info(f"SKIP: Recent Naive Bayes model exists (age: {model_age})")
                    logger.info("Use --force-training to override")
                    self.log_stage_end("Naive Bayes Training", True, f"Using existing model: {latest_model.name}")
                    return True
            
            # Run Naive Bayes training
            logger.info("EXECUTING: Running train_naive_bayes_model.py...")
            
            result = subprocess.run(
                [sys.executable, 'train_naive_bayes_model.py'],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("SUCCESS: Naive Bayes training completed")
                
                # Check if model files were created
                new_model_files = list(Path(self.nb_model_dir).glob('*.pkl'))
                
                if new_model_files:
                    latest_model = max(new_model_files, key=lambda p: p.stat().st_mtime)
                    self.log_stage_end("Naive Bayes Training", True, f"Model saved: {latest_model.name}")
                    self.stage_results['naive_bayes'] = {"model_path": str(latest_model)}
                    return True
                else:
                    self.log_stage_end("Naive Bayes Training", False, "No model files created")
                    return False
            else:
                error_msg = f"Naive Bayes training failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr[-1000:]}"
                
                self.log_stage_end("Naive Bayes Training", False, error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "Naive Bayes training timed out after 30 minutes"
            self.log_stage_end("Naive Bayes Training", False, error_msg)
            return False
            
        except Exception as e:
            error_msg = f"Naive Bayes training error: {e}"
            self.log_stage_end("Naive Bayes Training", False, error_msg)
            return False
    
    def run_lstm_training(self) -> bool:
        """Run the LSTM model training stage"""
        self.log_stage_start("LSTM Training")
        
        try:
            # Check if model already exists
            model_files = list(Path(self.lstm_model_dir).glob('*.h5'))
            
            if model_files:
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                model_age = datetime.now() - datetime.fromtimestamp(latest_model.stat().st_mtime)
                
                if model_age < timedelta(hours=12):  # Model is less than 12 hours old
                    logger.info(f"SKIP: Recent LSTM model exists (age: {model_age})")
                    logger.info("Use --force-training to override")
                    self.log_stage_end("LSTM Training", True, f"Using existing model: {latest_model.name}")
                    return True
            
            # Run LSTM training
            logger.info("EXECUTING: Running train_lstm_sentiment_model.py...")
            
            result = subprocess.run(
                [sys.executable, 'train_lstm_sentiment_model.py'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for LSTM training
            )
            
            if result.returncode == 0:
                logger.info("SUCCESS: LSTM training completed")
                
                # Check if model files were created
                new_model_files = list(Path(self.lstm_model_dir).glob('*.h5'))
                
                if new_model_files:
                    latest_model = max(new_model_files, key=lambda p: p.stat().st_mtime)
                    self.log_stage_end("LSTM Training", True, f"Model saved: {latest_model.name}")
                    self.stage_results['lstm'] = {"model_path": str(latest_model)}
                    return True
                else:
                    self.log_stage_end("LSTM Training", False, "No model files created")
                    return False
            else:
                error_msg = f"LSTM training failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr[-1000:]}"
                
                self.log_stage_end("LSTM Training", False, error_msg)
                return False
                
        except subprocess.TimeoutExpired:
            error_msg = "LSTM training timed out after 1 hour"
            self.log_stage_end("LSTM Training", False, error_msg)
            return False
            
        except Exception as e:
            error_msg = f"LSTM training error: {e}"
            self.log_stage_end("LSTM Training", False, error_msg)
            return False

    
    def generate_model_comparison(self):
        """Generate performance comparison between Naive Bayes and LSTM models"""
        try:
            logger.info("=" * 70)
            logger.info("MODEL PERFORMANCE COMPARISON")
            logger.info("=" * 70)
            
            # Load Naive Bayes results
            nb_metrics = self.load_model_metrics('naive_bayes')
            
            # Load LSTM results  
            lstm_metrics = self.load_model_metrics('lstm')
            
            # Display performance comparison
            self.display_performance_comparison(nb_metrics, lstm_metrics)
            
            # Save comparison data
            self.save_performance_comparison(nb_metrics, lstm_metrics)
            
        except Exception as e:
            logger.error(f"Failed to generate model comparison: {e}")

    def load_model_metrics(self, model_type: str) -> Dict:
        """Load performance metrics from model report files"""
        try:
            # Find the latest report file
            if model_type == 'naive_bayes':
                report_pattern = "*naive_bayes*training*report*.json"
            else:
                report_pattern = "*lstm*training*report*.json"
                
            report_files = list(Path('reports').glob(report_pattern))
            
            if not report_files:
                logger.warning(f"No {model_type} report files found")
                return {'status': 'No report found'}
            
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading {model_type} metrics from: {latest_report.name}")
            
            import json
            with open(latest_report, 'r') as f:
                data = json.load(f)
            
            # ✅ UNIFIED: Both models now have the same flat structure
            return {
                'accuracy': data.get('test_accuracy', 'N/A'),
                'f1_score': data.get('f1_score', self.extract_f1_from_report(data.get('classification_report', ''))),
                'precision': data.get('precision', self.extract_precision_from_report(data.get('classification_report', ''))),
                'recall': data.get('recall', self.extract_recall_from_report(data.get('classification_report', ''))),
                'status': 'available'
            }
            
        except Exception as e:
            logger.warning(f"Could not load {model_type} metrics: {e}")
            return {'status': f'Error loading metrics: {e}'}

    def extract_naive_bayes_metrics(self, data: Dict) -> Dict:
        """Extract Naive Bayes metrics from nested report structure"""
        try:
            # Check multiple possible locations for metrics
            metrics = {'status': 'available'}
            
            # Try tuned model first (best performance)
            if 'hyperparameter_tuning' in data:
                tuned = data['hyperparameter_tuning']
                metrics.update({
                    'accuracy': tuned.get('accuracy', 'N/A'),
                    'f1_score': tuned.get('f1_score', 'N/A'),
                    'precision': tuned.get('precision', 'N/A'),
                    'recall': tuned.get('recall', 'N/A')
                })
                logger.info("Using tuned model metrics for Naive Bayes")
                return metrics
            
            # Try best model from comparison
            if 'model_comparison' in data:
                comparison = data['model_comparison']
                # Find the best model (highest F1 score)
                best_model_name = None
                best_f1 = -1
                
                for model_name, model_data in comparison.items():
                    if isinstance(model_data, dict) and 'f1_score' in model_data:
                        f1 = model_data.get('f1_score', 0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_model_name = model_name
                
                if best_model_name:
                    best_model = comparison[best_model_name]
                    metrics.update({
                        'accuracy': best_model.get('accuracy', 'N/A'),
                        'f1_score': best_model.get('f1_score', 'N/A'),
                        'precision': best_model.get('precision', 'N/A'),
                        'recall': best_model.get('recall', 'N/A')
                    })
                    logger.info(f"Using best model metrics ({best_model_name}) for Naive Bayes")
                    return metrics
            
            # Try metadata section
            if 'metadata' in data and 'tuned_model' in data['metadata']:
                tuned_meta = data['metadata']['tuned_model']
                metrics.update({
                    'accuracy': tuned_meta.get('accuracy', 'N/A'),
                    'f1_score': tuned_meta.get('f1_score', 'N/A'),
                    'precision': 'N/A',  # Not in metadata
                    'recall': 'N/A'      # Not in metadata
                })
                logger.info("Using metadata metrics for Naive Bayes")
                return metrics
            
            logger.warning("Could not find metrics in Naive Bayes report structure")
            return {'status': 'Metrics not found in report'}
            
        except Exception as e:
            logger.error(f"Error extracting Naive Bayes metrics: {e}")
            return {'status': 'Error extracting metrics'}

    def extract_f1_from_report(self, report_text: str) -> float:
        """Extract weighted average F1 score from classification report"""
        try:
            import re
            lines = str(report_text).split('\n')
            for line in lines:
                if 'weighted avg' in line:
                    numbers = re.findall(r'\d+\.\d+', line)
                    if len(numbers) >= 3:
                        return float(numbers[2])
            return 'N/A'
        except:
            return 'N/A'

    def extract_precision_from_report(self, report_text: str) -> float:
        """Extract weighted average precision from classification report"""
        try:
            import re
            lines = str(report_text).split('\n')
            for line in lines:
                if 'weighted avg' in line:
                    numbers = re.findall(r'\d+\.\d+', line)
                    if len(numbers) >= 1:
                        return float(numbers[0])
            return 'N/A'
        except:
            return 'N/A'

    def extract_recall_from_report(self, report_text: str) -> float:
        """Extract weighted average recall from classification report"""
        try:
            import re
            lines = str(report_text).split('\n')
            for line in lines:
                if 'weighted avg' in line:
                    numbers = re.findall(r'\d+\.\d+', line)
                    if len(numbers) >= 2:
                        return float(numbers[1])
            return 'N/A'
        except:
            return 'N/A'

    def display_performance_comparison(self, nb_metrics: Dict, lstm_metrics: Dict):
        """Display performance comparison table"""
        logger.info("PERFORMANCE METRICS:")
        logger.info("-" * 25)
        logger.info(f"{'Metric':<12} {'Naive Bayes':<12} {'LSTM':<12} {'Winner':<10}")
        logger.info("-" * 50)
        
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        winners = {}
        
        for metric in metrics:
            nb_value = nb_metrics.get(metric, 'N/A')
            lstm_value = lstm_metrics.get(metric, 'N/A')
            
            # Determine winner
            winner = "N/A"
            if nb_value != 'N/A' and lstm_value != 'N/A':
                try:
                    nb_float = float(nb_value)
                    lstm_float = float(lstm_value)
                    if nb_float > lstm_float:
                        winner = "NB"
                        winners[metric] = 'naive_bayes'
                    elif lstm_float > nb_float:
                        winner = "LSTM"
                        winners[metric] = 'lstm'
                    else:
                        winner = "Tie"
                        winners[metric] = 'tie'
                except:
                    winner = "N/A"
            
            # Format values
            nb_display = f"{nb_value:.3f}" if isinstance(nb_value, (int, float)) else str(nb_value)
            lstm_display = f"{lstm_value:.3f}" if isinstance(lstm_value, (int, float)) else str(lstm_value)
            
            logger.info(f"{metric.title():<12} {nb_display:<12} {lstm_display:<12} {winner:<10}")
        
        logger.info("-" * 50)
        
        # Overall winner
        if winners:
            nb_wins = sum(1 for w in winners.values() if w == 'naive_bayes')
            lstm_wins = sum(1 for w in winners.values() if w == 'lstm')
            ties = sum(1 for w in winners.values() if w == 'tie')
            
            logger.info(f"WINS: Naive Bayes({nb_wins}) | LSTM({lstm_wins}) | Ties({ties})")
            
            if nb_wins > lstm_wins:
                logger.info("OVERALL WINNER: Naive Bayes")
            elif lstm_wins > nb_wins:
                logger.info("OVERALL WINNER: LSTM")
            else:
                logger.info("OVERALL RESULT: Tie")
        
        logger.info("=" * 70)

    def save_performance_comparison(self, nb_metrics: Dict, lstm_metrics: Dict):
        """Save performance comparison to JSON file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comparison_file = f"reports/performance_comparison_{timestamp}.json"
            
            comparison_data = {
                'timestamp': datetime.now().isoformat(),
                'naive_bayes': nb_metrics,
                'lstm': lstm_metrics,
                'pipeline_duration': str(datetime.now() - self.pipeline_start_time)
            }
            
            import json
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
            
            logger.info(f"Performance comparison saved to: {comparison_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance comparison: {e}")
    
    def run_pipeline(self) -> bool:
        """Run the complete batch pipeline"""
        logger.info("STARTING: Malaysian Tourism Sentiment Analysis Batch Pipeline")
        logger.info("=" * 70)
        logger.info(f"Pipeline Configuration:")
        logger.info(f"  Skip Data Collection: {self.skip_collection}")
        logger.info(f"  Skip Naive Bayes: {self.skip_naive_bayes}")
        logger.info(f"  Skip LSTM: {self.skip_lstm}")
        logger.info(f"  Data File: {self.data_file}")
        logger.info(f"  Minimum Samples: {self.min_data_samples}")
        logger.info("=" * 70)
        
        try:
            success = True
            
            # Stage 1: Data Collection
            if not self.skip_collection:
                if not self.run_data_collection():
                    logger.error("FAILED: Data collection stage failed")
                    success = False
            else:
                logger.info("SKIPPED: Data collection stage")
                
                # Still validate existing data
                if os.path.exists(self.data_file):
                    is_quality_ok, quality_stats = self.check_data_quality(self.data_file)
                    if is_quality_ok:
                        self.stage_results['data_collection'] = quality_stats
                        logger.info(f"VALIDATED: Existing data has {quality_stats['total_samples']} samples")
                    else:
                        logger.error(f"INVALID: Existing data has quality issues: {quality_stats.get('issues', quality_stats.get('error'))}")
                        success = False
                else:
                    logger.error("MISSING: No data file found and collection skipped")
                    success = False
            
            # Stage 2: Naive Bayes Training
            if success and not self.skip_naive_bayes:
                if not self.run_naive_bayes_training():
                    logger.error("FAILED: Naive Bayes training stage failed")
                    success = False
            else:
                logger.info("SKIPPED: Naive Bayes training stage")
            
            # Stage 3: LSTM Training
            if success and not self.skip_lstm:
                if not self.run_lstm_training():
                    logger.error("FAILED: LSTM training stage failed")
                    success = False
            else:
                logger.info("SKIPPED: LSTM training stage")
            
            # Generate final report
            self.generate_model_comparison()
            
            # Final status
            total_duration = datetime.now() - self.pipeline_start_time
            
            if success:
                logger.info("SUCCESS: Batch pipeline completed successfully!")
                logger.info(f"Total execution time: {total_duration}")
                logger.info("All stages completed. Models are ready for deployment.")
            else:
                logger.error("FAILED: Batch pipeline completed with errors!")
                logger.error(f"Total execution time: {total_duration}")
                logger.error("Check logs and reports for details.")
            
            return success
            
        except KeyboardInterrupt:
            logger.info("INTERRUPTED: Pipeline stopped by user")
            return False
            
        except Exception as e:
            logger.error(f"FATAL: Pipeline failed with unexpected error: {e}")
            logger.error(f"Full error: {traceback.format_exc()}")
            return False

def main():
    """Main execution function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Malaysian Tourism Sentiment Analysis Batch Pipeline')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection stage')
    parser.add_argument('--skip-naive-bayes', action='store_true', help='Skip Naive Bayes training stage')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM training stage')
    parser.add_argument('--force-collection', action='store_true', help='Force data collection even if recent data exists')
    parser.add_argument('--force-training', action='store_true', help='Force model training even if recent models exist')
    
    args = parser.parse_args()
    
    try:
        # Create and run pipeline
        pipeline = MalaysianTourismBatchPipeline(
            skip_collection=args.skip_collection,
            skip_naive_bayes=args.skip_naive_bayes,
            skip_lstm=args.skip_lstm
        )
        
        success = pipeline.run_pipeline()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"APPLICATION FAILED: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())