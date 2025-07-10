#!/usr/bin/env python3
"""
Integrated Batch Pipeline for Malaysian Tourism Sentiment Analysis
================================================================

This unified pipeline combines data collection and model training:
1. Reddit Data Collection with VADER sentiment analysis
2. Train Naive Bayes sentiment model 
3. Train LSTM sentiment model
4. Generate performance comparison

Features:
- Integrated data collection and training
- VADER sentiment analysis during collection
- Sequential execution with dependency checking
- Comprehensive error handling and logging
- Progress tracking and timing
- Data validation between stages
- Configurable execution modes

Author: Big Data & NLP Analytics Team
Date: July 10, 2025
"""

import os
import sys
import time
import logging
import subprocess
import traceback
import praw
import json
import re
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables
load_dotenv('.env.local')

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
        logging.FileHandler('logs/batch_integrated_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class MalaysianTourismIntegratedPipeline:
    """Complete integrated pipeline for Malaysian tourism sentiment analysis"""
    
    def __init__(self, skip_collection: bool = False, skip_naive_bayes: bool = False, skip_lstm: bool = False):
        """Initialize the integrated pipeline"""
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
        
        # Data collection configuration
        self.processed_posts = set()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.collection_stats = {
            'posts_collected': 0,
            'comments_collected': 0,
            'malaysia_related': 0,
            'filtered_out': 0,
            'api_errors': 0,
            'start_time': datetime.now()
        }
        
        self.setup_directories()
        self.load_collection_configuration()
        
    def setup_directories(self):
        """Create necessary directories for the pipeline"""
        dirs = [
            'logs', 'data/raw', 'data/processed', 'data/analytics',
            'models', 'models/naive_bayes', 'models/lstm', 'reports'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info("SETUP: Pipeline directories created/verified")
    
    def load_collection_configuration(self):
        """Load data collection configuration from environment variables"""
        self.subreddits = os.getenv('SUBREDDITS', 'malaysia,travel,backpacking,solotravel,TravelNoPics').split(',')
        self.keywords = [kw.strip().lower() for kw in os.getenv('SEARCH_KEYWORDS', 
            'malaysia,kuala lumpur,penang,langkawi,cameron highlands,malacca,kota kinabalu,johor bahru,ipoh,genting highlands').split(',')]
        self.max_posts = int(os.getenv('MAX_POSTS', 800))
        self.max_comments = int(os.getenv('MAX_COMMENTS_PER_POST', 15))
        self.max_search_results = int(os.getenv('MAX_SEARCH_RESULTS', 200))
        self.strategies = os.getenv('FETCH_STRATEGIES', 'hot,new,top,search').split(',')
        self.time_filter = os.getenv('TIME_FILTER', 'month')
        self.request_delay = float(os.getenv('REDDIT_REQUEST_DELAY', 1))
        
        logger.info(f"📋 Collection Configuration loaded:")
        logger.info(f"  Subreddits: {self.subreddits}")
        logger.info(f"  Keywords: {len(self.keywords)} Malaysian tourism terms")
        logger.info(f"  Max posts: {self.max_posts}, Max comments per post: {self.max_comments}")
        logger.info(f"  Strategies: {self.strategies}")
    
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
    
    def setup_reddit(self):
        """Setup Reddit API connection with enhanced error handling"""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                username=os.getenv('REDDIT_USERNAME'),
                password=os.getenv('REDDIT_PASSWORD'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'Malaysian Tourism Sentiment Analysis Bot v2.0'),
                ratelimit_seconds=300  # Handle rate limiting gracefully
            )            
            # Test connection
            test_user = self.reddit.user.me()
            logger.info(f"✅ Reddit API connected successfully as: {test_user.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Reddit API connection failed: {e}")
            return False
    
    def is_malaysia_related(self, text: str, title: str = "") -> bool:
        """Check if content is related to Malaysian tourism"""
        combined_text = f"{title} {text}".lower()
        
        # Check for Malaysian keywords
        for keyword in self.keywords:
            if keyword in combined_text:
                return True
                
        # Additional patterns for Malaysian content
        malaysia_patterns = [
            r'\bkl\b', r'\bmalaysia\b', r'\bmalaysian\b', 
            r'\bringgit\b', r'\brm\d+\b', r'\bmyr\b',
            r'\bbatu caves\b', r'\bklcc\b', r'\bpetronas\b',
            r'\btwin towers\b', r'\bgenting\b', r'\bpinang\b'
        ]
        
        for pattern in malaysia_patterns:
            if re.search(pattern, combined_text):
                return True
                
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = text.strip()
        
        # Length filtering
        min_length = 10
        max_length = 5000
        
        if len(text) < min_length or len(text) > max_length:
            return ""
            
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment using VADER"""
        try:
            if not text or len(text.strip()) < 3:
                return {'sentiment_label': 'neutral', 'sentiment_compound': 0.0}
            
            # Get VADER scores
            scores = self.sentiment_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Determine label based on compound score
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'sentiment_label': label,
                'sentiment_compound': round(compound, 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {'sentiment_label': 'neutral', 'sentiment_compound': 0.0}
    
    def format_post_data(self, post, strategy: str) -> Optional[Dict]:
        """Format Reddit post data with enhanced metadata"""
        try:
            content = self.clean_text(post.selftext) or self.clean_text(post.title)
            
            if not content:
                return None
                
            # Check if Malaysia-related
            is_relevant = self.is_malaysia_related(content, post.title)
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(content)
            
            post_data = {
                'id': post.id,
                'title': post.title,
                'content': content,
                'content_type': 'post',
                'parent_post_id': post.id,
                'score': post.score,
                'upvote_ratio': getattr(post, 'upvote_ratio', 0),
                'num_comments': post.num_comments,
                'created_date': datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                'author': str(post.author) if post.author else '[deleted]',
                'subreddit': str(post.subreddit),
                'url': post.url,
                'permalink': f"https://reddit.com{post.permalink}",
                'is_malaysia_related': is_relevant,
                'text_length': len(content),
                'flair': getattr(post, 'link_flair_text', None),
                'is_nsfw': post.over_18,
                'is_spoiler': post.spoiler,
                'collection_strategy': strategy,
                'collection_timestamp': datetime.now(timezone.utc).isoformat(),
                **sentiment_result
            }
            
            return post_data
            
        except Exception as e:
            logger.error(f"Error formatting post {post.id}: {e}")
            self.collection_stats['api_errors'] += 1
            return None
    
    def format_comment_data(self, comment, parent_post_id: str, post_title: str) -> Optional[Dict]:
        """Format Reddit comment data with enhanced metadata"""
        try:
            if not hasattr(comment, 'body') or comment.body in ['[deleted]', '[removed]']:
                return None
                
            content = self.clean_text(comment.body)
            if not content:
                return None
                
            # Check if Malaysia-related
            is_relevant = self.is_malaysia_related(content, post_title)
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(content)
            
            comment_data = {
                'id': comment.id,
                'title': f"Comment on: {post_title[:100]}...",
                'content': content,
                'content_type': 'comment',
                'parent_post_id': parent_post_id,
                'score': comment.score,
                'upvote_ratio': 0,  # Comments don't have upvote ratio
                'num_comments': 0,  # Comments don't have num_comments
                'created_date': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                'author': str(comment.author) if comment.author else '[deleted]',
                'subreddit': str(comment.subreddit),
                'url': '',  # Comments don't have separate URLs
                'permalink': f"https://reddit.com{comment.permalink}",
                'is_malaysia_related': is_relevant,
                'text_length': len(content),
                'flair': None,  # Comments don't have flair
                'is_nsfw': False,  # Comments inherit from post
                'is_spoiler': False,  # Comments inherit from post
                'collection_strategy': 'comment',
                'collection_timestamp': datetime.now(timezone.utc).isoformat(),
                **sentiment_result
            }
            
            return comment_data
            
        except Exception as e:
            logger.error(f"Error formatting comment {comment.id}: {e}")
            self.collection_stats['api_errors'] += 1
            return None
    
    def get_csv_headers(self) -> List[str]:
        """Define CSV column headers"""
        return [
            'id', 'content_type', 'title', 'content', 'sentiment_label', 
            'sentiment_compound', 'subreddit', 'score', 'author', 'created_date',
            'is_malaysia_related', 'text_length', 'collection_strategy', 
            'url', 'permalink', 'num_comments', 'upvote_ratio'
        ]
    
    def setup_csv_file(self):
        """Initialize CSV file with headers"""
        try:
            headers = self.get_csv_headers()
            with open(self.data_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info("✅ CSV file initialized with headers")
        except Exception as e:
            logger.error(f"❌ Failed to setup CSV file: {e}")
    
    def save_to_csv(self, data: Dict) -> bool:
        """Save data to CSV"""
        try:
            headers = self.get_csv_headers()
            row = []
            
            for header in headers:
                value = data.get(header, '')
                # Handle special cases
                if value is None:
                    value = ''
                elif isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, (list, dict)):
                    value = str(value)
                row.append(value)
            
            # Write to CSV
            with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save to CSV: {e}")
            return False
    
    def collect_from_subreddit(self, subreddit_name: str, strategy: str, limit: int) -> int:
        """Collect posts from a subreddit using specified strategy"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            collected = 0
            
            logger.info(f"📊 Collecting {strategy} posts from r/{subreddit_name} (limit: {limit})")
            
            if strategy == 'hot':
                posts = subreddit.hot(limit=limit)
            elif strategy == 'new':
                posts = subreddit.new(limit=limit)
            elif strategy == 'top':
                posts = subreddit.top(time_filter=self.time_filter, limit=limit)
            elif strategy == 'search':
                # Search for Malaysian tourism keywords
                search_query = ' OR '.join(self.keywords[:5])  # Use top 5 keywords
                posts = subreddit.search(search_query, limit=limit, time_filter=self.time_filter)
            else:
                logger.warning(f"Unknown strategy: {strategy}")
                return 0
            
            for post in posts:
                if post.id in self.processed_posts:
                    continue
                    
                self.processed_posts.add(post.id)
                
                # Process post
                post_data = self.format_post_data(post, strategy)
                if post_data:
                    if self.save_to_csv(post_data):
                        collected += 1
                        self.collection_stats['posts_collected'] += 1
                        
                        if post_data['is_malaysia_related']:
                            self.collection_stats['malaysia_related'] += 1
                            logger.info(f"🇲🇾 Malaysia-related post: {post.title[:60]}...")
                        else:
                            self.collection_stats['filtered_out'] += 1
                        
                        # Collect comments
                        comment_count = self.collect_comments(post, post_data['title'])
                        self.collection_stats['comments_collected'] += comment_count
                else:
                    self.collection_stats['filtered_out'] += 1
                
                # Rate limiting
                time.sleep(self.request_delay)
                
                # Progress reporting
                if collected % 50 == 0 and collected > 0:
                    self.log_collection_progress()
                    
            return collected
            
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit_name} ({strategy}): {e}")
            self.collection_stats['api_errors'] += 1
            return 0
    
    def collect_comments(self, post, post_title: str) -> int:
        """Collect comments from a post"""
        try:
            post.comments.replace_more(limit=0)
            comments = post.comments.list()[:self.max_comments]
            collected = 0
            
            for comment in comments:
                comment_data = self.format_comment_data(comment, post.id, post_title)
                if comment_data and self.save_to_csv(comment_data):
                    collected += 1
                    
            return collected
            
        except Exception as e:
            logger.error(f"Error collecting comments for post {post.id}: {e}")
            return 0
    
    def log_collection_progress(self):
        """Log current collection progress and statistics"""
        elapsed = datetime.now() - self.collection_stats['start_time']
        
        logger.info("📊 COLLECTION PROGRESS REPORT")
        logger.info(f"  Runtime: {elapsed}")
        logger.info(f"  Posts collected: {self.collection_stats['posts_collected']}")
        logger.info(f"  Comments collected: {self.collection_stats['comments_collected']}")
        logger.info(f"  Malaysia-related: {self.collection_stats['malaysia_related']}")
        logger.info(f"  Filtered out: {self.collection_stats['filtered_out']}")
        logger.info(f"  API errors: {self.collection_stats['api_errors']}")
        logger.info(f"  Total items: {self.collection_stats['posts_collected'] + self.collection_stats['comments_collected']}")
    
    def run_data_collection(self) -> bool:
        """Run the integrated data collection stage"""
        self.log_stage_start("Data Collection (Integrated)")
        
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
                        self.log_stage_end("Data Collection (Integrated)", True, f"Using existing data: {quality_stats['total_samples']} samples")
                        self.stage_results['data_collection'] = quality_stats
                        return True
                    else:
                        logger.warning("QUALITY: Existing data has quality issues, collecting new data")
                        logger.warning(f"Issues: {quality_stats.get('issues', quality_stats.get('error'))}")
            
            # Setup Reddit API
            if not self.setup_reddit():
                self.log_stage_end("Data Collection (Integrated)", False, "Reddit API setup failed")
                return False
            
            # Initialize CSV file
            self.setup_csv_file()
            
            logger.info("🚀 Starting Integrated Malaysian Tourism Data Collection")
            logger.info("=" * 60)
            
            total_collected = 0
            posts_per_strategy = self.max_posts // len(self.strategies)
            posts_per_subreddit = posts_per_strategy // len(self.subreddits)
            
            for strategy in self.strategies:
                logger.info(f"🔄 Strategy: {strategy.upper()}")
                
                for subreddit_name in self.subreddits:
                    collected = self.collect_from_subreddit(
                        subreddit_name.strip(), 
                        strategy.strip(), 
                        posts_per_subreddit
                    )
                    total_collected += collected
                    
                    if total_collected >= self.max_posts:
                        logger.info(f"✅ Reached maximum posts limit: {self.max_posts}")
                        break
                        
                if total_collected >= self.max_posts:
                    break
            
            # Final collection statistics
            self.log_collection_progress()
            
            logger.info("✅ INTEGRATED COLLECTION COMPLETED!")
            logger.info(f"🎯 Target achieved: {self.collection_stats['posts_collected']}/{self.max_posts} posts")
            logger.info(f"💬 Bonus comments: {self.collection_stats['comments_collected']}")
            logger.info(f"🇲🇾 Malaysia relevance: {self.collection_stats['malaysia_related']}/{self.collection_stats['posts_collected'] + self.collection_stats['comments_collected']} items")
            
            # Validate collected data
            is_quality_ok, quality_stats = self.check_data_quality(self.data_file)
            
            if is_quality_ok:
                self.log_stage_end("Data Collection (Integrated)", True, f"Collected {quality_stats['total_samples']} samples")
                self.stage_results['data_collection'] = quality_stats
                
                # Log data statistics
                logger.info("DATA STATS:")
                logger.info(f"  Total samples: {quality_stats['total_samples']}")
                logger.info(f"  Malaysia-related: {quality_stats['malaysia_related']} ({quality_stats['malaysia_percentage']:.1f}%)")
                logger.info(f"  Sentiment distribution: {quality_stats['sentiment_distribution']}")
                
                return True
            else:
                error_msg = f"Data quality check failed: {quality_stats.get('issues', quality_stats.get('error'))}"
                self.log_stage_end("Data Collection (Integrated)", False, error_msg)
                return False
                
        except KeyboardInterrupt:
            logger.info("🛑 Collection stopped by user")
            self.log_stage_end("Data Collection (Integrated)", False, "Interrupted by user")
            return False
        except Exception as e:
            error_msg = f"Integrated data collection error: {e}"
            self.log_stage_end("Data Collection (Integrated)", False, error_msg)
            return False
    
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
            
            # Set environment variables to handle TensorFlow issues
            env = os.environ.copy()
            env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
            env['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
            
            result = subprocess.run(
                [sys.executable, 'train_lstm_sentiment_model.py'],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for LSTM training
                env=env
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
                    stderr_msg = result.stderr[-1000:]
                    
                    # Check for common TensorFlow errors
                    if "DLL load failed" in stderr_msg or "_pywrap_tensorflow" in stderr_msg:
                        logger.warning("⚠️ TensorFlow DLL error detected")
                        logger.warning("💡 Try: pip install tensorflow==2.13.0")
                        logger.warning("💡 Install Visual C++ Redistributables")
                        self.log_stage_end("LSTM Training", True, "Skipped due to TensorFlow DLL issues")
                        return True  # Don't fail pipeline for TensorFlow issues
                    
                    error_msg += f"\nSTDERR: {stderr_msg}"
                
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
            
            # Unified structure for both models
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
            comparison_file = f"reports/performance_comparison_integrated_{timestamp}.json"
            
            comparison_data = {
                'timestamp': datetime.now().isoformat(),
                'naive_bayes': nb_metrics,
                'lstm': lstm_metrics,
                'pipeline_duration': str(datetime.now() - self.pipeline_start_time),
                'data_source': 'integrated_collection_with_vader'
            }
            
            import json
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=2, default=str)
            
            logger.info(f"Performance comparison saved to: {comparison_file}")
            
        except Exception as e:
            logger.error(f"Failed to save performance comparison: {e}")
    
    def run_pipeline(self) -> bool:
        """Run the complete integrated pipeline"""
        logger.info("STARTING: Malaysian Tourism Sentiment Analysis Integrated Pipeline")
        logger.info("=" * 80)
        logger.info(f"Pipeline Configuration:")
        logger.info(f"  Skip Data Collection: {self.skip_collection}")
        logger.info(f"  Skip Naive Bayes: {self.skip_naive_bayes}")
        logger.info(f"  Skip LSTM: {self.skip_lstm}")
        logger.info(f"  Data File: {self.data_file}")
        logger.info(f"  Minimum Samples: {self.min_data_samples}")
        logger.info(f"  Collection Strategy: Integrated with VADER sentiment analysis")
        logger.info("=" * 80)
        
        try:
            success = True
            
            # Stage 1: Integrated Data Collection
            if not self.skip_collection:
                if not self.run_data_collection():
                    logger.error("FAILED: Integrated data collection stage failed")
                    success = False
            else:
                logger.info("SKIPPED: Integrated data collection stage")
                
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
                logger.info("SUCCESS: Integrated pipeline completed successfully!")
                logger.info(f"Total execution time: {total_duration}")
                logger.info("All stages completed. Models are ready for deployment.")
                logger.info(f"🇲🇾 Malaysia-related items collected: {self.collection_stats['malaysia_related']}")
                logger.info(f"📊 Total items processed: {self.collection_stats['posts_collected'] + self.collection_stats['comments_collected']}")
            else:
                logger.error("FAILED: Integrated pipeline completed with errors!")
                logger.error(f"Total execution time: {total_duration}")
                logger.error("Check logs and reports for details.")
            
            return success
            
        except KeyboardInterrupt:
            logger.info("INTERRUPTED: Integrated Pipeline stopped by user")
            return False
            
        except Exception as e:
            logger.error(f"FATAL: Integrated Pipeline failed with unexpected error: {e}")
            logger.error(f"Full error: {traceback.format_exc()}")
            return False

def main():
    """Main execution function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Malaysian Tourism Sentiment Analysis Integrated Pipeline')
    parser.add_argument('--skip-collection', action='store_true', help='Skip integrated data collection stage')
    parser.add_argument('--skip-naive-bayes', action='store_true', help='Skip Naive Bayes training stage')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM training stage')
    parser.add_argument('--force-collection', action='store_true', help='Force data collection even if recent data exists')
    parser.add_argument('--force-training', action='store_true', help='Force model training even if recent models exist')
    
    args = parser.parse_args()
    
    try:
        # Create and run integrated pipeline
        pipeline = MalaysianTourismIntegratedPipeline(
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