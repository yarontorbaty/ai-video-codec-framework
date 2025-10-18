"""
V3.0 S3 Uploader

Handles uploading reconstructed videos and decoder code to S3
"""

import boto3
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# S3 Configuration
BUCKET_NAME = os.getenv('S3_BUCKET', 'ai-codec-v3-artifacts-580473065386')
REGION = os.getenv('AWS_REGION', 'us-east-1')


class S3Uploader:
    """Upload experiment artifacts to S3"""
    
    def __init__(self):
        self.s3 = boto3.client('s3', region_name=REGION)
        self.bucket = BUCKET_NAME
    
    def upload_video(self, local_path: str, experiment_id: str) -> Optional[str]:
        """
        Upload reconstructed video to S3
        
        Returns:
            Presigned URL (valid for 7 days) or None if failed
        """
        try:
            s3_key = f"videos/{experiment_id}/reconstructed.mp4"
            
            logger.info(f"📤 Uploading video to s3://{self.bucket}/{s3_key}")
            
            self.s3.upload_file(
                local_path,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': 'video/mp4',
                    'Metadata': {
                        'experiment_id': experiment_id
                    }
                }
            )
            
            # Generate presigned URL (7 days expiration)
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': s3_key
                },
                ExpiresIn=7 * 24 * 3600  # 7 days
            )
            
            logger.info(f"✅ Video uploaded successfully")
            return url
            
        except Exception as e:
            logger.error(f"❌ Video upload failed: {e}", exc_info=True)
            return None
    
    def save_decoder(self, decoder_code: str, experiment_id: str) -> Optional[str]:
        """
        Save decoder code to S3
        
        Returns:
            S3 key or None if failed
        """
        try:
            s3_key = f"decoders/{experiment_id}/decoder.py"
            
            logger.info(f"📤 Saving decoder to s3://{self.bucket}/{s3_key}")
            
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=decoder_code.encode('utf-8'),
                ContentType='text/x-python',
                Metadata={
                    'experiment_id': experiment_id
                }
            )
            
            logger.info(f"✅ Decoder saved successfully")
            return s3_key
            
        except Exception as e:
            logger.error(f"❌ Decoder save failed: {e}", exc_info=True)
            return None
    
    def save_encoder(self, encoder_code: str, experiment_id: str) -> Optional[str]:
        """
        Save encoder code to S3
        
        Returns:
            S3 key or None if failed
        """
        try:
            s3_key = f"encoders/{experiment_id}/encoder.py"
            
            logger.info(f"📤 Saving encoder to s3://{self.bucket}/{s3_key}")
            
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=encoder_code.encode('utf-8'),
                ContentType='text/x-python',
                Metadata={
                    'experiment_id': experiment_id
                }
            )
            
            logger.info(f"✅ Encoder saved successfully")
            return s3_key
            
        except Exception as e:
            logger.error(f"❌ Encoder save failed: {e}", exc_info=True)
            return None

