"""
Malaysian Tourism Kafka Consumer with Real-time Sentiment Analysis
================================================================

This consumer processes tourism-related posts and comments from Kafka streams,
performing sentiment analysis and generating insights for Malaysian tourism.

Features:
- Real-time Kafka message consumption
- Multi-model sentiment analysis (TextBlob, VADER, Custom)
- Malaysian tourism-specific sentiment scoring
- Data enrichment and processing
- Real-time analytics and monitoring
- Comprehensive error handling and logging
- Dashboard-ready data output

Author: Big Data & NLP Analytics Team
Date: June 22, 2025
Course Deadline: June 27, 2025
"""

import json
import time
import logging
import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
import pandas as pd
from collections import defaultdict, deque
import threading
from pathlib import Path

# NLP and Sentiment Analysis imports
try:
    import nltk
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    NLP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NLP libraries not available: {e}")
    NLP_AVAILABLE = False

# Try to import Kafka
try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Kafka not available: {e}")
    print("📝 Running in FILE-ONLY mode - will process saved JSON files")
    KAFKA_AVAILABLE = False
    KafkaConsumer = None

# Load environment variables
load_dotenv('.env.local')

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/reddit_tourism_consumer.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class MalaysianTourismSentimentAnalyzer:
    """Advanced sentiment analysis for Malaysian tourism content"""
    
    def __init__(self):
        """Initialize sentiment analysis models"""
        self.setup_models()
        self.setup_malaysian_context()
        
    def setup_models(self):
        """Setup sentiment analysis models"""
        if NLP_AVAILABLE:
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                pass
                
            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()
            logger.info("✅ Sentiment analysis models initialized")
        else:
            logger.warning("⚠️ NLP libraries not available - using basic sentiment analysis")
            self.vader = None
    
    def setup_malaysian_context(self):
        """Setup Malaysian tourism-specific sentiment context"""
        # Positive Malaysian tourism keywords
        self.positive_keywords = {
            'amazing', 'beautiful', 'stunning', 'incredible', 'wonderful', 'fantastic',
            'delicious', 'friendly', 'helpful', 'clean', 'safe', 'affordable',
            'recommend', 'must-visit', 'love', 'enjoy', 'perfect', 'excellent',
            'cheap', 'budget-friendly', 'value', 'worth', 'authentic', 'cultural'
        }
        
        # Negative Malaysian tourism keywords
        self.negative_keywords = {
            'expensive', 'overpriced', 'dirty', 'crowded', 'tourist-trap', 'scam',
            'disappointed', 'terrible', 'awful', 'bad', 'worst', 'avoid',
            'unsafe', 'rude', 'unfriendly', 'boring', 'waste', 'overrated'
        }
        
        # Malaysian context boosters
        self.malaysian_boosters = {
            'nasi lemak': 0.3, 'rendang': 0.3, 'char kway teow': 0.2,
            'petronas towers': 0.4, 'batu caves': 0.3, 'langkawi': 0.4,
            'penang': 0.4, 'malacca': 0.3, 'cameron highlands': 0.3,
            'truly asia': 0.5, 'malaysia boleh': 0.4
        }
    
    def analyze_sentiment(self, text: str, title: str = "") -> Dict:
        """Perform comprehensive sentiment analysis"""
        combined_text = f"{title} {text}".lower()
        
        results = {
            'text_length': len(text),
            'title_length': len(title),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if NLP_AVAILABLE and self.vader:
            # VADER Sentiment
            vader_scores = self.vader.polarity_scores(text)
            results['vader'] = vader_scores
            
            # TextBlob Sentiment
            try:
                blob = TextBlob(text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")
                results['textblob'] = {'polarity': 0.0, 'subjectivity': 0.0}
        else:
            # Basic sentiment analysis
            results['vader'] = {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
            results['textblob'] = {'polarity': 0.0, 'subjectivity': 0.0}
        
        # Malaysian tourism-specific analysis
        malaysian_sentiment = self.analyze_malaysian_context(combined_text)
        results['malaysian_tourism'] = malaysian_sentiment
        
        # Combined sentiment score
        results['final_sentiment'] = self.calculate_final_sentiment(results)
        
        return results
    
    def analyze_malaysian_context(self, text: str) -> Dict:
        """Analyze sentiment with Malaysian tourism context"""
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        
        # Calculate Malaysian-specific boost
        malaysian_boost = 0.0
        for term, boost in self.malaysian_boosters.items():
            if term in text:
                malaysian_boost += boost
        
        # Calculate context-aware sentiment
        if positive_count + negative_count > 0:
            sentiment_ratio = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            sentiment_ratio = 0.0
        
        # Apply Malaysian boost
        final_score = sentiment_ratio + (malaysian_boost * 0.1)
        final_score = max(-1.0, min(1.0, final_score))  # Clamp to [-1, 1]
        
        return {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'malaysian_boost': malaysian_boost,
            'sentiment_score': final_score,
            'confidence': min(1.0, (positive_count + negative_count + malaysian_boost) / 10)
        }
    
    def calculate_final_sentiment(self, results: Dict) -> Dict:
        """Calculate final weighted sentiment score"""
        # Get individual scores
        vader_score = results['vader']['compound']
        textblob_score = results['textblob']['polarity']
        malaysian_score = results['malaysian_tourism']['sentiment_score']
        
        # Weighted combination
        weights = {
            'vader': 0.4,
            'textblob': 0.3,
            'malaysian': 0.3
        }
        
        final_score = (
            vader_score * weights['vader'] +
            textblob_score * weights['textblob'] +
            malaysian_score * weights['malaysian']
        )
        
        # Classify sentiment
        if final_score >= 0.1:
            sentiment_label = 'positive'
        elif final_score <= -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence
        confidence = (
            abs(final_score) * 0.7 +
            results['malaysian_tourism']['confidence'] * 0.3
        )
        
        return {
            'score': final_score,
            'label': sentiment_label,
            'confidence': min(1.0, confidence),
            'weights_used': weights
        }

class MalaysianTourismConsumer:
    """Enhanced Kafka consumer for Malaysian tourism sentiment analysis"""
    
    def __init__(self):
        """Initialize consumer with sentiment analysis capabilities"""
        self.setup_directories()
        self.setup_sentiment_analyzer()
        self.setup_consumer()
        self.load_configuration()
        self.setup_analytics()
        self.running = False
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['logs', 'data/processed', 'data/analytics', 'data/dashboard']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        logger.info("✅ Consumer directories created/verified")
    
    def setup_sentiment_analyzer(self):
        """Initialize sentiment analyzer"""
        self.sentiment_analyzer = MalaysianTourismSentimentAnalyzer()
        logger.info("✅ Sentiment analyzer initialized")
    
    def setup_consumer(self):
        """Setup Kafka consumer or file processing mode"""
        if not KAFKA_AVAILABLE:
            logger.info("⚠️ Kafka not available - using file processing mode")
            self.consumer = None
            self.file_mode = True
            return
        
        try:
            self.consumer = KafkaConsumer(
                os.getenv('KAFKA_TOPIC', 'reddit-malaysia-tourism'),
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
                group_id=os.getenv('KAFKA_CONSUMER_GROUP', 'malaysia-sentiment-group'),
                auto_offset_reset=os.getenv('KAFKA_AUTO_OFFSET_RESET', 'earliest'),
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=10000  # 10 second timeout
            )
            self.file_mode = False
            logger.info(f"✅ Kafka consumer connected to topic: {os.getenv('KAFKA_TOPIC')}")
            
        except Exception as e:
            logger.error(f"❌ Kafka consumer setup failed: {e}")
            logger.info("📝 Falling back to file processing mode")
            self.consumer = None
            self.file_mode = True
    
    def load_configuration(self):
        """Load configuration from environment variables"""
        self.batch_size = int(os.getenv('CONSUMER_BATCH_SIZE', 100))
        self.analytics_interval = int(os.getenv('ANALYTICS_INTERVAL', 60))  # seconds
        self.dashboard_update_interval = int(os.getenv('DASHBOARD_UPDATE_INTERVAL', 300))  # 5 minutes
        
        # Output file configuration
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.processed_file = f'data/processed/sentiment_analysis_{timestamp}.jsonl'
        self.analytics_file = f'data/analytics/tourism_sentiment_report_{timestamp}.json'
        self.dashboard_file = f'data/dashboard/realtime_dashboard_data.json'
        
        logger.info(f"📋 Consumer configuration loaded:")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Analytics interval: {self.analytics_interval}s")
        logger.info(f"  Output file: {self.processed_file}")
    
    def setup_analytics(self):
        """Setup analytics tracking"""
        self.stats = {
            'messages_processed': 0,
            'posts_analyzed': 0,
            'comments_analyzed': 0,
            'sentiment_distribution': defaultdict(int),
            'malaysia_related_count': 0,
            'processing_errors': 0,
            'start_time': datetime.now(),
            'last_analytics_update': datetime.now()
        }
        
        # Recent sentiment trends (sliding window)
        self.recent_sentiments = deque(maxlen=1000)
        self.hourly_sentiment = defaultdict(list)
        
        logger.info("✅ Analytics tracking initialized")
    
    def process_message(self, message: Dict) -> Optional[Dict]:
        """Process individual message with sentiment analysis"""
        try:
            # Extract text content
            content = message.get('content', '')
            title = message.get('title', '')
            
            if not content:
                return None
            
            # Perform sentiment analysis
            sentiment_results = self.sentiment_analyzer.analyze_sentiment(content, title)
            
            # Enrich message with sentiment data
            enriched_message = message.copy()
            enriched_message['sentiment_analysis'] = sentiment_results
            enriched_message['processing_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Add to analytics
            self.update_analytics(enriched_message)
            
            return enriched_message
            
        except Exception as e:
            logger.error(f"Error processing message {message.get('id', 'unknown')}: {e}")
            self.stats['processing_errors'] += 1
            return None
    
    def update_analytics(self, processed_message: Dict):
        """Update analytics with processed message"""
        self.stats['messages_processed'] += 1
        
        # Count content types
        if processed_message.get('content_type') == 'post':
            self.stats['posts_analyzed'] += 1
        elif processed_message.get('content_type') == 'comment':
            self.stats['comments_analyzed'] += 1
        
        # Track Malaysia-related content
        if processed_message.get('is_malaysia_related', False):
            self.stats['malaysia_related_count'] += 1
        
        # Track sentiment distribution
        sentiment_label = processed_message['sentiment_analysis']['final_sentiment']['label']
        self.stats['sentiment_distribution'][sentiment_label] += 1
        
        # Add to recent sentiments for trend analysis
        sentiment_score = processed_message['sentiment_analysis']['final_sentiment']['score']
        timestamp = datetime.now()
        self.recent_sentiments.append({
            'timestamp': timestamp,
            'score': sentiment_score,
            'label': sentiment_label
        })
        
        # Hourly aggregation
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')
        self.hourly_sentiment[hour_key].append(sentiment_score)
    
    def save_processed_message(self, processed_message: Dict):
        """Save processed message to file"""
        try:
            with open(self.processed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(processed_message, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save processed message: {e}")
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        current_time = datetime.now()
        runtime = current_time - self.stats['start_time']
        
        # Calculate sentiment trends
        if self.recent_sentiments:
            recent_scores = [s['score'] for s in self.recent_sentiments]
            avg_sentiment = sum(recent_scores) / len(recent_scores)
            sentiment_trend = self.calculate_sentiment_trend()
        else:
            avg_sentiment = 0.0
            sentiment_trend = 'stable'
        
        # Calculate hourly averages
        hourly_averages = {}
        for hour, scores in self.hourly_sentiment.items():
            if scores:
                hourly_averages[hour] = sum(scores) / len(scores)
        
        report = {
            'timestamp': current_time.isoformat(),
            'runtime': str(runtime),
            'processing_stats': dict(self.stats),
            'sentiment_summary': {
                'total_analyzed': self.stats['messages_processed'],
                'malaysia_related_percentage': (
                    self.stats['malaysia_related_count'] / max(1, self.stats['messages_processed']) * 100
                ),
                'average_sentiment': avg_sentiment,
                'sentiment_trend': sentiment_trend,
                'distribution': dict(self.stats['sentiment_distribution'])
            },
            'hourly_sentiment_averages': hourly_averages,
            'processing_rate': {
                'messages_per_minute': self.stats['messages_processed'] / max(1, runtime.total_seconds() / 60),
                'error_rate': self.stats['processing_errors'] / max(1, self.stats['messages_processed']) * 100
            }
        }
        
        return report
    
    def calculate_sentiment_trend(self) -> str:
        """Calculate sentiment trend from recent data"""
        if len(self.recent_sentiments) < 10:
            return 'insufficient_data'
        
        # Compare first and second half of recent sentiments
        half_point = len(self.recent_sentiments) // 2
        first_half_avg = sum(s['score'] for s in list(self.recent_sentiments)[:half_point]) / half_point
        second_half_avg = sum(s['score'] for s in list(self.recent_sentiments)[half_point:]) / (len(self.recent_sentiments) - half_point)
        
        difference = second_half_avg - first_half_avg
        
        if difference > 0.1:
            return 'improving'
        elif difference < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def save_analytics_report(self):
        """Save analytics report to file"""
        try:
            report = self.generate_analytics_report()
            
            # Save detailed report
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Save dashboard data (simplified for real-time display)
            dashboard_data = {
                'last_updated': datetime.now().isoformat(),
                'total_processed': self.stats['messages_processed'],
                'malaysia_related': self.stats['malaysia_related_count'],
                'sentiment_distribution': dict(self.stats['sentiment_distribution']),
                'average_sentiment': report['sentiment_summary']['average_sentiment'],
                'sentiment_trend': report['sentiment_summary']['sentiment_trend'],
                'processing_rate': report['processing_rate']['messages_per_minute']
            }
            
            with open(self.dashboard_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"📊 Analytics report saved: {self.analytics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analytics report: {e}")
    
    def log_progress(self):
        """Log current processing progress"""
        runtime = datetime.now() - self.stats['start_time']
        
        logger.info("📊 PROCESSING PROGRESS")
        logger.info(f"  Runtime: {runtime}")
        logger.info(f"  Messages processed: {self.stats['messages_processed']}")
        logger.info(f"  Posts: {self.stats['posts_analyzed']}, Comments: {self.stats['comments_analyzed']}")
        logger.info(f"  Malaysia-related: {self.stats['malaysia_related_count']}")
        logger.info(f"  Sentiment distribution: {dict(self.stats['sentiment_distribution'])}")
        logger.info(f"  Processing errors: {self.stats['processing_errors']}")
    
    def process_from_files(self):
        """Process messages from JSON files when Kafka is not available"""
        logger.info("📁 Processing from saved files...")
        
        # Find JSON files in data/raw directory
        raw_data_dir = Path('data/raw')
        if not raw_data_dir.exists():
            logger.error("❌ No data/raw directory found")
            return
        
        json_files = list(raw_data_dir.glob('*.jsonl'))
        if not json_files:
            logger.error("❌ No JSONL files found in data/raw directory")
            return
        
        logger.info(f"📂 Found {len(json_files)} files to process")
        
        processed_count = 0
        for file_path in json_files:
            logger.info(f"📄 Processing file: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            message = json.loads(line.strip())
                            processed_message = self.process_message(message)
                            
                            if processed_message:
                                self.save_processed_message(processed_message)
                                processed_count += 1
                                
                                # Progress reporting
                                if processed_count % 50 == 0:
                                    self.log_progress()
                                    
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in {file_path.name} line {line_num}: {e}")
                        except Exception as e:
                            logger.error(f"Error processing line {line_num} in {file_path.name}: {e}")
                            
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        logger.info(f"✅ File processing completed! Processed {processed_count} messages")
    
    def run_kafka_consumer(self):
        """Run Kafka consumer loop"""
        logger.info("🚀 Starting Kafka consumer...")
        
        try:
            processed_batch = []
            
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    processed_message = self.process_message(message.value)
                    
                    if processed_message:
                        processed_batch.append(processed_message)
                        
                        # Process in batches
                        if len(processed_batch) >= self.batch_size:
                            self.save_batch(processed_batch)
                            processed_batch = []
                    
                    # Periodic analytics update
                    if (datetime.now() - self.stats['last_analytics_update']).total_seconds() > self.analytics_interval:
                        self.save_analytics_report()
                        self.log_progress()
                        self.stats['last_analytics_update'] = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    self.stats['processing_errors'] += 1
            
            # Save any remaining messages in batch
            if processed_batch:
                self.save_batch(processed_batch)
                
        except KeyboardInterrupt:
            logger.info("🛑 Consumer stopped by user")
        except Exception as e:
            logger.error(f"❌ Kafka consumer error: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
    
    def save_batch(self, batch: List[Dict]):
        """Save a batch of processed messages"""
        for message in batch:
            self.save_processed_message(message)
        logger.info(f"💾 Saved batch of {len(batch)} processed messages")
    
    def run(self):
        """Main consumer execution"""
        logger.info("🚀 Starting Malaysian Tourism Sentiment Consumer")
        logger.info("=" * 60)
        
        self.running = True
        
        try:
            if self.file_mode:
                # Process from files
                self.process_from_files()
            else:
                # Start analytics reporting thread
                analytics_thread = threading.Thread(
                    target=self.periodic_analytics_update,
                    daemon=True
                )
                analytics_thread.start()
                
                # Run Kafka consumer
                self.run_kafka_consumer()
            
            # Final analytics report
            self.save_analytics_report()
            self.log_progress()
            
            logger.info("✅ SENTIMENT ANALYSIS COMPLETED!")
            logger.info(f"📊 Total processed: {self.stats['messages_processed']} messages")
            logger.info(f"🇲🇾 Malaysia-related: {self.stats['malaysia_related_count']}")
            logger.info(f"📈 Sentiment distribution: {dict(self.stats['sentiment_distribution'])}")
            logger.info(f"📁 Results saved to: {self.processed_file}")
            logger.info(f"📊 Analytics report: {self.analytics_file}")
            
        except Exception as e:
            logger.error(f"❌ Consumer execution failed: {e}")
            raise
        finally:
            self.running = False
            logger.info("🔚 Consumer shutdown complete")
    
    def periodic_analytics_update(self):
        """Periodic analytics update in separate thread"""
        while self.running:
            time.sleep(self.dashboard_update_interval)
            if self.running:  # Check again after sleep
                self.save_analytics_report()

def main():
    """Main execution function"""
    try:
        consumer = MalaysianTourismConsumer()
        consumer.run()
        
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())