#!/usr/bin/env python3
"""
Admin API for AiV1 Codec - Chat with Governing LLM and Control Experiments
"""

import json
import boto3
import os
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
ssm = boto3.client('ssm', region_name='us-east-1')
s3 = boto3.client('s3', region_name='us-east-1')
secretsmanager = boto3.client('secretsmanager', region_name='us-east-1')

# Tables
experiments_table = dynamodb.Table('ai-video-codec-experiments')
control_table = dynamodb.Table('ai-video-codec-control')  # New table for control state

# Instance ID
ORCHESTRATOR_INSTANCE_ID = 'i-063947ae46af6dbf8'

# Admin password hash (set via environment variable)
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '')

# Fetch API key from Secrets Manager
ANTHROPIC_API_KEY = None
try:
    secret_response = secretsmanager.get_secret_value(SecretId='ai-video-codec/anthropic-api-key')
    secret_data = json.loads(secret_response['SecretString'])
    ANTHROPIC_API_KEY = secret_data.get('ANTHROPIC_API_KEY')
    logger.info("✅ API key retrieved from Secrets Manager")
except Exception as e:
    logger.error(f"Failed to retrieve API key from Secrets Manager: {e}")


def verify_admin(username, password):
    """Verify admin credentials."""
    # Fetch credentials from Secrets Manager
    try:
        secret_response = secretsmanager.get_secret_value(SecretId='ai-video-codec/admin-credentials')
        secret_data = json.loads(secret_response['SecretString'])
        stored_username = secret_data.get('username')
        stored_password = secret_data.get('password')
        
        if not stored_username or not stored_password:
            logger.error("Admin credentials not configured in Secrets Manager")
            return False
        
        # Verify credentials
        return username == stored_username and password == stored_password
    except Exception as e:
        logger.error(f"Failed to verify credentials: {e}")
        return False


def create_session_token(username):
    """Create a session token."""
    import secrets
    import time
    
    token = secrets.token_urlsafe(32)
    expiry = int(time.time()) + (24 * 60 * 60)  # 24 hours
    
    # Store in DynamoDB
    try:
        control_table.put_item(Item={
            'control_id': f'session_{token}',
            'username': username,
            'expiry': expiry,
            'created_at': datetime.utcnow().isoformat()
        })
        return token
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return None


def verify_session_token(token):
    """Verify a session token."""
    import time
    
    try:
        response = control_table.get_item(Key={'control_id': f'session_{token}'})
        item = response.get('Item')
        
        if not item:
            return False
        
        # Check expiry
        expiry = int(item.get('expiry', 0))
        if time.time() > expiry:
            # Delete expired session
            control_table.delete_item(Key={'control_id': f'session_{token}'})
            return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to verify session: {e}")
        return False


def send_2fa_code(email, code):
    """Send 2FA code via email using AWS SES."""
    try:
        ses = boto3.client('ses', region_name='us-east-1')
        
        response = ses.send_email(
            Source=email,  # Must be verified in SES
            Destination={'ToAddresses': [email]},
            Message={
                'Subject': {
                    'Data': f'Your AiV1 verification code is {code}',
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Text': {
                        'Data': f'''
AiV1 Admin Login Verification

Your verification code is: {code}

This code will expire in 10 minutes.

If you didn't request this code, you can safely ignore this email.

---
AiV1 Autonomous Video Codec Project
                        ''',
                        'Charset': 'UTF-8'
                    },
                    'Html': {
                        'Data': f'''
                            <!DOCTYPE html>
                            <html lang="en">
                            <head>
                                <meta charset="UTF-8">
                                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                <title>Your Verification Code</title>
                            </head>
                            <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f5f5f5;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5; padding: 20px;">
                                    <tr>
                                        <td align="center">
                                            <table width="600" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                                <tr>
                                                    <td style="padding: 40px 30px; text-align: center;">
                                                        <h1 style="margin: 0 0 20px 0; color: #667eea; font-size: 24px; font-weight: 600;">AiV1 Admin Login</h1>
                                                        <p style="margin: 0 0 30px 0; color: #333; font-size: 16px; line-height: 1.5;">
                                                            Your verification code is:
                                                        </p>
                                                        <div style="background-color: #667eea; color: white; font-size: 36px; font-weight: bold; padding: 20px; border-radius: 8px; letter-spacing: 8px; margin: 0 auto 30px auto; display: inline-block;">
                                                            {code}
                                                        </div>
                                                        <p style="margin: 0 0 10px 0; color: #666; font-size: 14px;">
                                                            This code will expire in 10 minutes.
                                                        </p>
                                                        <p style="margin: 0; color: #666; font-size: 14px;">
                                                            If you didn't request this code, you can safely ignore this email.
                                                        </p>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td style="padding: 20px 30px; background-color: #f8f9fa; text-align: center; border-top: 1px solid #e0e0e0;">
                                                        <p style="margin: 0; color: #999; font-size: 12px;">
                                                            AiV1 Autonomous Video Codec Project
                                                        </p>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </body>
                            </html>
                        ''',
                        'Charset': 'UTF-8'
                    }
                }
            }
        )
        logger.info(f"2FA code sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send 2FA email: {e}")
        return False


def generate_2fa_code():
    """Generate a 6-digit 2FA code."""
    import random
    return str(random.randint(100000, 999999))


def handle_login(username, password):
    """Handle login request - send 2FA code."""
    if verify_admin(username, password):
        # Generate and store 2FA code
        code = generate_2fa_code()
        expiry = int(datetime.utcnow().timestamp()) + (10 * 60)  # 10 minutes
        
        # Get email from secrets
        try:
            secret_response = secretsmanager.get_secret_value(SecretId='ai-video-codec/admin-credentials')
            secret_data = json.loads(secret_response['SecretString'])
            email = secret_data.get('email')
            twofa_enabled = secret_data.get('2fa_enabled', False)
            
            if not twofa_enabled or not email:
                # 2FA not configured - skip and login directly
                token = create_session_token(username)
                if token:
                    return {
                        "success": True,
                        "token": token,
                        "username": username,
                        "requires_2fa": False,
                        "message": "Login successful"
                    }
                else:
                    return {"success": False, "error": "Failed to create session"}
            
            # Store 2FA code
            control_table.put_item(Item={
                'control_id': f'2fa_{username}',
                'code': code,
                'expiry': expiry,
                'created_at': datetime.utcnow().isoformat()
            })
            
            # Send email
            if send_2fa_code(email, code):
                return {
                    "success": True,
                    "requires_2fa": True,
                    "message": "2FA code sent to your email",
                    "username": username
                }
            else:
                return {"success": False, "error": "Failed to send 2FA code"}
                
        except Exception as e:
            logger.error(f"Error handling 2FA: {e}")
            return {"success": False, "error": "Authentication error"}
    else:
        return {"success": False, "error": "Invalid credentials"}


def verify_2fa_and_login(username, code):
    """Verify 2FA code and create session."""
    try:
        # Get stored code
        response = control_table.get_item(Key={'control_id': f'2fa_{username}'})
        item = response.get('Item')
        
        if not item:
            return {"success": False, "error": "2FA code expired or not found"}
        
        # Check expiry
        expiry = int(item.get('expiry', 0))
        if datetime.utcnow().timestamp() > expiry:
            # Delete expired code
            control_table.delete_item(Key={'control_id': f'2fa_{username}'})
            return {"success": False, "error": "2FA code expired"}
        
        # Verify code
        stored_code = item.get('code')
        if code != stored_code:
            return {"success": False, "error": "Invalid 2FA code"}
        
        # Delete used code
        control_table.delete_item(Key={'control_id': f'2fa_{username}'})
        
        # Create session
        token = create_session_token(username)
        if token:
            return {
                "success": True,
                "token": token,
                "username": username,
                "message": "Login successful"
            }
        else:
            return {"success": False, "error": "Failed to create session"}
            
    except Exception as e:
        logger.error(f"Error verifying 2FA: {e}")
        return {"success": False, "error": "Verification failed"}


def call_llm_chat(message, history):
    """
    Call governing LLM with admin message.
    """
    try:
        import urllib.request
        import urllib.error
        
        if not ANTHROPIC_API_KEY:
            return {"error": "LLM API key not configured - check Secrets Manager"}
        
        # Build conversation with system prompt
        messages = []
        
        # Add history
        for msg in history[-10:]:  # Last 10 messages
            role = "user" if msg['role'] == 'user' else "assistant"
            messages.append({"role": role, "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # System prompt
        system_prompt = """You are the Governing LLM for the AiV1 autonomous video codec project. 
You are chatting with the human administrator who can override your decisions.

Your role:
1. Answer questions about the current state of experiments
2. Provide insights and suggestions for new approaches
3. Explain your reasoning and hypotheses
4. Accept guidance and suggestions from the admin
5. Generate commands when requested (format: COMMAND: {command_name})

Available commands:
- START_EXPERIMENT: Start a new experiment
- STOP_EXPERIMENTS: Stop all running experiments
- PAUSE_AUTONOMOUS: Pause autonomous mode
- RESUME_AUTONOMOUS: Resume autonomous mode
- SUGGEST_CONFIG: Suggest a new configuration

Current project status: Experiments running, parameter storage achieving 0.04 Mbps, 
LLM code generation active. Latest insight: codec architecture is inverted."""

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = json.dumps({
            "model": "claude-sonnet-4-5",
            "max_tokens": 2048,
            "temperature": 0.7,
            "system": system_prompt,
            "messages": messages
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {
                "response": result['content'][0]['text'],
                "commands": extract_commands(result['content'][0]['text'])
            }
    
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "error": str(e),
            "response": f"Error communicating with LLM: {str(e)}"
        }


def extract_commands(text):
    """Extract commands from LLM response."""
    commands = []
    lines = text.split('\n')
    
    for line in lines:
        if line.strip().startswith('COMMAND:'):
            cmd = line.replace('COMMAND:', '').strip()
            commands.append({
                "command": cmd,
                "description": cmd
            })
    
    return commands


def get_system_status():
    """Get current system status."""
    try:
        # Get total experiments
        response = experiments_table.scan(Select='COUNT')
        total_experiments = response.get('Count', 0)
        
        # Get recent experiments for stats
        response = experiments_table.scan(Limit=50)
        experiments = response.get('Items', [])
        
        # Calculate stats
        best_bitrate = None
        success_count = 0
        
        for exp in experiments:
            if exp.get('status') == 'completed':
                success_count += 1
                
                # Parse experiments field
                exp_data = exp.get('experiments', '[]')
                if isinstance(exp_data, str):
                    exp_data = json.loads(exp_data)
                
                for e in exp_data:
                    metrics = e.get('real_metrics', {})
                    bitrate = metrics.get('bitrate_mbps')
                    if bitrate and (best_bitrate is None or bitrate < best_bitrate):
                        best_bitrate = float(bitrate)
        
        success_rate = (success_count / total_experiments * 100) if total_experiments > 0 else 0
        
        # Check if experiments are running
        running_now = check_running_experiments()
        
        # Check autonomous mode
        control_item = control_table.get_item(Key={'control_id': 'autonomous_mode'}).get('Item', {})
        autonomous_enabled = control_item.get('enabled', True)
        
        return {
            "total_experiments": total_experiments,
            "running_now": running_now,
            "best_bitrate": best_bitrate,
            "success_rate": success_rate,
            "autonomous_enabled": autonomous_enabled,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"error": str(e)}


def check_running_experiments():
    """Check if experiments are currently running."""
    try:
        # Check if orchestrator instance is running
        response = ssm.describe_instance_information(
            Filters=[{'Key': 'InstanceIds', 'Values': [ORCHESTRATOR_INSTANCE_ID]}]
        )
        
        if response.get('InstanceInformationList'):
            # Check recent command executions
            commands = ssm.list_commands(
                InstanceId=ORCHESTRATOR_INSTANCE_ID,
                MaxResults=5
            )
            
            for cmd in commands.get('Commands', []):
                if cmd['Status'] == 'InProgress':
                    return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"Error checking running experiments: {e}")
        return 0


def start_experiment_command():
    """Start a new experiment."""
    try:
        response = ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'cd /opt/ai-video-codec',
                    'aws s3 cp s3://ai-video-codec-videos-580473065386/framework/ai-codec-framework.tar.gz .',
                    'tar -xzf ai-codec-framework.tar.gz',
                    'source /root/.bashrc',
                    'python3 scripts/real_experiment.py'
                ]
            },
            TimeoutSeconds=600
        )
        
        return {
            "success": True,
            "message": f"Experiment started. Command ID: {response['Command']['CommandId']}"
        }
    
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        return {"success": False, "message": str(e)}


def stop_experiments_command():
    """Stop all running experiments."""
    try:
        # List running commands
        commands = ssm.list_commands(
            InstanceId=ORCHESTRATOR_INSTANCE_ID,
            MaxResults=10
        )
        
        stopped = 0
        for cmd in commands.get('Commands', []):
            if cmd['Status'] == 'InProgress':
                ssm.cancel_command(
                    CommandId=cmd['CommandId'],
                    InstanceIds=[ORCHESTRATOR_INSTANCE_ID]
                )
                stopped += 1
        
        return {
            "success": True,
            "message": f"Stopped {stopped} running experiments"
        }
    
    except Exception as e:
        logger.error(f"Error stopping experiments: {e}")
        return {"success": False, "message": str(e)}


def pause_autonomous_command():
    """Pause autonomous mode."""
    try:
        control_table.put_item(Item={
            'control_id': 'autonomous_mode',
            'enabled': False,
            'updated_at': datetime.utcnow().isoformat()
        })
        
        return {"success": True, "message": "Autonomous mode paused"}
    
    except Exception as e:
        logger.error(f"Error pausing autonomous: {e}")
        return {"success": False, "message": str(e)}


def resume_autonomous_command():
    """Resume autonomous mode."""
    try:
        control_table.put_item(Item={
            'control_id': 'autonomous_mode',
            'enabled': True,
            'updated_at': datetime.utcnow().isoformat()
        })
        
        return {"success": True, "message": "Autonomous mode resumed"}
    
    except Exception as e:
        logger.error(f"Error resuming autonomous: {e}")
        return {"success": False, "message": str(e)}


def handle_command(command):
    """Execute admin command."""
    command_map = {
        'start_experiment': start_experiment_command,
        'stop_experiments': stop_experiments_command,
        'pause_autonomous': pause_autonomous_command,
        'resume_autonomous': resume_autonomous_command,
    }
    
    handler = command_map.get(command.lower())
    if handler:
        return handler()
    else:
        return {"success": False, "message": f"Unknown command: {command}"}


def lambda_handler(event, context):
    """Main Lambda handler for admin API."""
    
    try:
        # Parse request
        path = event.get('path', '')
        method = event.get('httpMethod', 'GET')
        
        # CORS headers
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
        }
        
        # Handle OPTIONS for CORS
        if method == 'OPTIONS':
            return {'statusCode': 200, 'headers': headers, 'body': '{}'}
        
        # Parse body
        body = {}
        if event.get('body'):
            body = json.loads(event['body'])
        
        # Authentication check (except for login and 2FA verification endpoints)
        if path not in ['/admin/login', '/admin/verify-2fa']:
            # Check for Authorization header
            auth_header = event.get('headers', {}).get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return {
                    'statusCode': 401,
                    'headers': headers,
                    'body': json.dumps({"error": "Unauthorized - missing or invalid token"})
                }
            
            # Verify token
            token = auth_header.replace('Bearer ', '')
            if not verify_session_token(token):
                return {
                    'statusCode': 401,
                    'headers': headers,
                    'body': json.dumps({"error": "Unauthorized - invalid or expired token"})
                }
        
        # Route requests
        if path == '/admin/login' and method == 'POST':
            username = body.get('username', '')
            password = body.get('password', '')
            response_body = handle_login(username, password)
        
        elif path == '/admin/verify-2fa' and method == 'POST':
            username = body.get('username', '')
            code = body.get('code', '')
            response_body = verify_2fa_and_login(username, code)
        
        elif path == '/admin/status':
            response_body = get_system_status()
        
        elif path == '/admin/chat' and method == 'POST':
            message = body.get('message', '')
            history = body.get('history', [])
            response_body = call_llm_chat(message, history)
        
        elif path == '/admin/command' and method == 'POST':
            command = body.get('command', '')
            response_body = handle_command(command)
        
        elif path == '/admin/execute' and method == 'POST':
            command = body.get('command', '')
            response_body = handle_command(command)
        
        else:
            response_body = {"error": "Not found"}
            return {
                'statusCode': 404,
                'headers': headers,
                'body': json.dumps(response_body)
            }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_body, default=str)
        }
    
    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({"error": str(e)})
        }

