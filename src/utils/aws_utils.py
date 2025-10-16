#!/usr/bin/env python3
"""
AWS Utilities for AI Video Codec Framework
Handles S3 operations, DynamoDB interactions, and CloudWatch metrics.
"""

import boto3
import logging
import json
from typing import Dict, List, Optional, Any
from botocore.exceptions import ClientError
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class AWSUtils:
    """AWS utilities for the AI codec framework."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
    def download_from_s3(self, bucket: str, key: str, local_path: str) -> bool:
        """Download file from S3."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {key} from {bucket} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Error downloading {key} from {bucket}: {e}")
            return False
    
    def upload_to_s3(self, local_path: str, bucket: str, key: str, metadata: Optional[Dict] = None) -> bool:
        """Upload file to S3."""
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
                
            self.s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading {local_path} to {bucket}/{key}: {e}")
            return False
    
    def list_s3_objects(self, bucket: str, prefix: str = '') -> List[Dict]:
        """List objects in S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            return response.get('Contents', [])
        except ClientError as e:
            logger.error(f"Error listing objects in {bucket}: {e}")
            return []
    
    def put_dynamodb_item(self, table_name: str, item: Dict) -> bool:
        """Put item in DynamoDB table."""
        try:
            table = self.dynamodb.Table(table_name)
            table.put_item(Item=item)
            logger.info(f"Put item in {table_name}: {item}")
            return True
        except ClientError as e:
            logger.error(f"Error putting item in {table_name}: {e}")
            return False
    
    def get_dynamodb_item(self, table_name: str, key: Dict) -> Optional[Dict]:
        """Get item from DynamoDB table."""
        try:
            table = self.dynamodb.Table(table_name)
            response = table.get_item(Key=key)
            return response.get('Item')
        except ClientError as e:
            logger.error(f"Error getting item from {table_name}: {e}")
            return None
    
    def update_dynamodb_item(self, table_name: str, key: Dict, updates: Dict) -> bool:
        """Update item in DynamoDB table."""
        try:
            table = self.dynamodb.Table(table_name)
            
            # Build update expression
            update_expression = "SET "
            expression_attribute_values = {}
            expression_attribute_names = {}
            
            for i, (attr, value) in enumerate(updates.items()):
                if i > 0:
                    update_expression += ", "
                update_expression += f"#{attr} = :{attr}"
                expression_attribute_names[f"#{attr}"] = attr
                expression_attribute_values[f":{attr}"] = value
            
            table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values
            )
            logger.info(f"Updated item in {table_name}: {key}")
            return True
        except ClientError as e:
            logger.error(f"Error updating item in {table_name}: {e}")
            return False
    
    def put_cloudwatch_metric(self, namespace: str, metric_name: str, value: float, 
                             unit: str = 'None', dimensions: Optional[Dict] = None) -> bool:
        """Put custom metric to CloudWatch."""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow()
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )
            logger.info(f"Put metric {metric_name} = {value} to CloudWatch")
            return True
        except ClientError as e:
            logger.error(f"Error putting metric to CloudWatch: {e}")
            return False
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment status from DynamoDB."""
        return self.get_dynamodb_item('ai-video-codec-experiments', {'experiment_id': experiment_id})
    
    def update_experiment_status(self, experiment_id: str, status: str, 
                               metrics: Optional[Dict] = None) -> bool:
        """Update experiment status in DynamoDB."""
        updates = {
            'status': status,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        if metrics:
            updates['metrics'] = json.dumps(metrics)
        
        return self.update_dynamodb_item(
            'ai-video-codec-experiments',
            {'experiment_id': experiment_id},
            updates
        )
    
    def log_experiment_metrics(self, experiment_id: str, metrics: Dict) -> bool:
        """Log experiment metrics to CloudWatch and DynamoDB."""
        # Log to CloudWatch
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.put_cloudwatch_metric(
                    'AI-Video-Codec',
                    metric_name,
                    value,
                    dimensions={'ExperimentId': experiment_id}
                )
        
        # Update DynamoDB
        return self.update_experiment_status(experiment_id, 'running', metrics)
    
    def create_experiment_record(self, experiment_id: str, config: Dict) -> bool:
        """Create new experiment record in DynamoDB."""
        item = {
            'experiment_id': experiment_id,
            'status': 'started',
            'config': json.dumps(config),
            'created_at': datetime.utcnow().isoformat(),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return self.put_dynamodb_item('ai-video-codec-experiments', item)
