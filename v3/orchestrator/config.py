"""
V3.0 Configuration

Centralized configuration for orchestrator
"""

import os
import json
import boto3
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management"""
    
    def __init__(self):
        # AWS Configuration
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.dynamodb_table = os.getenv('DYNAMODB_TABLE', 'ai-codec-v3-experiments')
        
        # Worker Configuration
        self.worker_url = os.getenv('WORKER_URL', 'http://10.0.2.10:8080')
        
        # Orchestrator Configuration
        self.max_iterations = int(os.getenv('MAX_ITERATIONS', '100'))
        self.iteration_delay_sec = int(os.getenv('ITERATION_DELAY_SEC', '60'))
        
        # Load Anthropic API key from Secrets Manager
        self.anthropic_api_key = self._load_api_key()
    
    def _load_api_key(self) -> str:
        """Load Anthropic API key from AWS Secrets Manager"""
        try:
            secret_name = os.getenv('ANTHROPIC_SECRET', 'ai-video-codec/anthropic-api-key')
            
            logger.info(f"🔑 Loading API key from Secrets Manager: {secret_name}")
            
            client = boto3.client('secretsmanager', region_name=self.region)
            response = client.get_secret_value(SecretId=secret_name)
            
            secret = json.loads(response['SecretString'])
            api_key = secret.get('ANTHROPIC_API_KEY')
            
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in secret")
            
            logger.info(f"✅ API key loaded successfully")
            return api_key
            
        except Exception as e:
            logger.error(f"❌ Failed to load API key: {e}")
            # Fallback to environment variable
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                logger.info("✅ Using API key from environment variable")
                return api_key
            else:
                raise ValueError("No Anthropic API key found")

