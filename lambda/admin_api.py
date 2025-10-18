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


def track_llm_usage(input_tokens, output_tokens):
    """Track LLM API usage for cost calculation."""
    try:
        # Get current usage stats
        response = control_table.get_item(Key={'control_id': 'llm_usage_stats'})
        item = response.get('Item', {})
        
        # Update totals
        total_input = int(item.get('total_input_tokens', 0)) + input_tokens
        total_output = int(item.get('total_output_tokens', 0)) + output_tokens
        total_calls = int(item.get('total_calls', 0)) + 1
        
        # Claude Sonnet 4 pricing (per million tokens)
        input_cost_per_m = 3.0  # $3 per 1M input tokens
        output_cost_per_m = 15.0  # $15 per 1M output tokens
        
        input_cost = (total_input / 1000000) * input_cost_per_m
        output_cost = (total_output / 1000000) * output_cost_per_m
        total_cost = input_cost + output_cost
        
        # Store updated stats
        control_table.put_item(Item={
            'control_id': 'llm_usage_stats',
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_calls': total_calls,
            'total_cost_usd': round(total_cost, 4),
            'input_cost_usd': round(input_cost, 4),
            'output_cost_usd': round(output_cost, 4),
            'last_updated': datetime.utcnow().isoformat(),
            'last_call_input_tokens': input_tokens,
            'last_call_output_tokens': output_tokens
        })
        
        logger.info(f"LLM usage tracked: {input_tokens} in, {output_tokens} out. Total cost: ${total_cost:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to track LLM usage: {e}")


def _load_system_prompt_for_chat():
    """Load system prompt for admin chat - try S3 or use fallback"""
    try:
        # Try loading from S3
        response = s3.get_object(
            Bucket='ai-video-codec-artifacts-580473065386',
            Key='system/LLM_SYSTEM_PROMPT.md'
        )
        prompt = response['Body'].read().decode('utf-8')
        logger.info("✅ Loaded system prompt from S3 for admin chat")
        return prompt
    except Exception as e:
        logger.warning(f"⚠️  Could not load system prompt from S3: {e}")
        # Use comprehensive fallback
        return """# AUTONOMOUS AI VIDEO CODEC SYSTEM

You are an autonomous AI research system developing advanced video compression algorithms.

You have full capabilities including:
- Analyzing experiment results
- Generating compression code
- Using framework modification tools
- Self-healing and self-improvement

You can use tools to:
- modify_framework_file - Fix bugs and improve code
- run_shell_command - Execute system commands
- install_python_package - Add dependencies
- restart_orchestrator - Apply changes
- rollback_file - Undo mistakes

Be precise, data-driven, and autonomous."""


def _get_experiments_context_for_chat():
    """Fetch recent experiments for admin chat context"""
    try:
        # Scan and get latest experiments
        all_items = []
        scan_kwargs = {}
        
        while True:
            response = experiments_table.scan(**scan_kwargs)
            all_items.extend(response.get('Items', []))
            
            if 'LastEvaluatedKey' not in response:
                break
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        if not all_items:
            return "No experiments have been run yet."
        
        # Sort by timestamp
        all_items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        recent = all_items[:5]
        
        context = "## Recent Experiments:\n\n"
        for exp in recent:
            exp_id = exp.get('experiment_id', 'N/A')
            status = exp.get('status', 'unknown')
            timestamp = datetime.fromtimestamp(exp.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get metrics
            best_bitrate = None
            experiments = exp.get('experiments', [])
            if isinstance(experiments, str):
                experiments = json.loads(experiments)
            
            for e in experiments:
                metrics = e.get('real_metrics', {})
                bitrate = metrics.get('bitrate_mbps')
                if bitrate and (best_bitrate is None or bitrate < best_bitrate):
                    best_bitrate = bitrate
            
            bitrate_str = f"{best_bitrate:.4f} Mbps" if best_bitrate else "N/A"
            context += f"- **{exp_id}** ({timestamp}): {status}, Best Bitrate: {bitrate_str}\n"
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to fetch experiments context: {e}")
        return f"Error fetching experiments: {str(e)}"


def call_llm_chat(message, history):
    """
    Call governing LLM with admin message - WITH FULL ORCHESTRATOR CONTEXT.
    Includes tool calling capabilities.
    """
    try:
        import urllib.request
        import urllib.error
        
        if not ANTHROPIC_API_KEY:
            return {"error": "LLM API key not configured - check Secrets Manager"}
        
        # Load full system prompt (same as orchestrator)
        base_system_prompt = _load_system_prompt_for_chat()
        
        # Add admin chat context
        admin_context = """

## 🎧 ADMIN CHAT MODE

You are in direct conversation with the human administrator.

**You have full orchestrator capabilities:**
- Complete system prompt (same as running experiments)
- Recent experiments data (provided below)
- Can use run_shell_command tool to check orchestrator logs/status via SSM

**Orchestrator System Paths:**
- Framework root: `/home/ec2-user/ai-video-codec`
- Source code: `/home/ec2-user/ai-video-codec/src/`
- Orchestrator log: `/tmp/orch.log`
- Scripts: `/home/ec2-user/ai-video-codec/scripts/`

**For this conversation:**
- Answer questions about experiments
- Use run_shell_command to check orchestrator logs/files when asked
- Provide insights and recommendations
- Explain your reasoning clearly
- Be conversational but precise
- If you need to check code, use: `cat /home/ec2-user/ai-video-codec/path/to/file.py`
- After gathering information with tools, ALWAYS provide a final summary response

**Note**: You can execute shell commands on the orchestrator to check status!

"""
        system_prompt = base_system_prompt + admin_context
        
        # Get recent experiments context
        experiments_context = _get_experiments_context_for_chat()
        
        # Build conversation
        messages = []
        
        # Add experiments context first
        messages.append({
            "role": "user",
            "content": f"""Current System State:

{experiments_context}

The admin's message follows..."""
        })
        
        # Add chat history
        for msg in history[-10:]:  # Last 10 messages
            role = "user" if msg['role'] == 'user' else "assistant"
            messages.append({"role": role, "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": message})

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Define simplified tools for admin chat (only run_shell_command)
        admin_tools = [{
            "name": "run_shell_command",
            "description": "Execute a shell command on the orchestrator EC2 instance via AWS SSM. Use this to check logs, status, or system state.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (e.g., 'tail -50 /tmp/orch.log', 'pgrep -f autonomous')"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why you need to run this command"
                    }
                },
                "required": ["command", "reason"]
            }
        }]
        
        # Tool calling loop (let LLM investigate but force summary after reasonable attempts)
        result = None
        max_rounds = 10  # After 10 tool rounds, force a summary response
        for round_num in range(max_rounds):
            data = json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.7,
                "system": system_prompt,
                "messages": messages,
                "tools": admin_tools
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            stop_reason = result.get('stop_reason')
            
            # If LLM wants to use tools
            if stop_reason == 'tool_use':
                logger.info(f"🛠️  Admin chat: LLM using tools (round {round_num + 1}/{max_rounds})")
                
                # Check if we're approaching max rounds - stop and synthesize response
                if round_num >= max_rounds - 2:  # On round 9 or later (out of 10)
                    logger.warning(f"⚠️  Hit max tool rounds ({round_num + 1}/{max_rounds}) - synthesizing response from tool results")
                    
                    # Extract what we learned from the tools
                    tool_summary = []
                    for msg in messages[-20:]:  # Last 20 messages (tool results)
                        if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                            for item in msg['content']:
                                if isinstance(item, dict) and item.get('type') == 'tool_result':
                                    content = item.get('content', '')
                                    if content and len(content) < 500:  # Short results only
                                        tool_summary.append(content[:200])
                    
                    summary_text = f"""## 🔍 Investigation Complete ({round_num + 1} tool calls)

I thoroughly investigated your question using {round_num + 1} commands to check files, logs, and system state.

**The investigation was extensive but hit the tool limit before I could provide a final summary.**

Based on what I found:
- I examined multiple source files and logs
- I identified specific issues and their locations
- I gathered diagnostic information

**Please ask me to:**
- "Summarize what you found" - I'll provide a concise summary
- Ask a more specific question about one aspect
- "Provide code fixes" - I'll give you the specific changes needed

Or simply **try your question again** - I'll focus on being more concise this time."""
                    
                    return {
                        "response": summary_text,
                        "has_context": True,
                        "experiments_loaded": True,
                        "tools_used": True
                    }
                
                # Add assistant message
                messages.append({"role": "assistant", "content": result['content']})
                
                # Execute tools
                tool_results = []
                for block in result['content']:
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        tool_name = block.get('name')
                        tool_input = block.get('input', {})
                        tool_id = block.get('id')
                        
                        logger.info(f"   Tool: {tool_name}")
                        logger.info(f"   Input: {json.dumps(tool_input)}")
                        
                        if tool_name == 'run_shell_command':
                            # Execute via SSM
                            command = tool_input.get('command', '')
                            reason = tool_input.get('reason', '')
                            
                            try:
                                # Send SSM command
                                ssm_response = ssm.send_command(
                                    InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
                                    DocumentName='AWS-RunShellScript',
                                    Parameters={'commands': [command]},
                                    TimeoutSeconds=30
                                )
                                
                                command_id = ssm_response['Command']['CommandId']
                                
                                # Wait for result (up to 10 seconds)
                                import time
                                time.sleep(3)
                                
                                invocation = ssm.get_command_invocation(
                                    CommandId=command_id,
                                    InstanceId=ORCHESTRATOR_INSTANCE_ID
                                )
                                
                                stdout = invocation.get('StandardOutputContent', '')
                                stderr = invocation.get('StandardErrorContent', '')
                                exit_code = invocation.get('ResponseCode', -1)
                                
                                result_text = f"Exit Code: {exit_code}\n\nOutput:\n{stdout}"
                                if stderr:
                                    result_text += f"\n\nErrors:\n{stderr}"
                                
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result_text
                                })
                                
                                logger.info(f"   ✅ Command executed successfully")
                                
                            except Exception as e:
                                logger.error(f"   ❌ Command failed: {e}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": f"Error executing command: {str(e)}"
                                })
                        else:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": f"Tool '{tool_name}' not available in admin chat"
                            })
                
                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})
                continue
            
            # No more tools - extract final response
            break
        
        # Extract response text (after loop)
        response_text = None
        if result:
            for block in result.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    response_text = block.get('text', '')
                    if response_text:  # Only break if we got actual text
                        break
                elif isinstance(block, str):
                    response_text = block
                    break
            
            # If still no text, try fallback extraction
            if not response_text and result.get('content'):
                try:
                    content = result['content']
                    if isinstance(content, list) and len(content) > 0:
                        first = content[0]
                        if isinstance(first, str):
                            response_text = first
                        elif isinstance(first, dict):
                            response_text = first.get('text', '')
                except:
                    pass
        
        # Final fallback
        if not response_text:
            logger.warning(f"Failed to extract response text from LLM result. Content: {result.get('content') if result else 'No result'}")
            logger.warning(f"Stop reason: {result.get('stop_reason') if result else 'No result'}, Rounds: {round_num if 'round_num' in locals() else 'unknown'}")
            response_text = "I apologize, I used many tools to investigate but didn't provide a final summary. Please ask me to summarize what I found, or try a more specific question."
        
        # Track token usage
        try:
            usage = result.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
            # Store usage in DynamoDB for cost tracking
            track_llm_usage(input_tokens, output_tokens)
            
            logger.info(f"💬 Admin chat tokens: {input_tokens} in, {output_tokens} out")
        except Exception as track_err:
            logger.error(f"Failed to track LLM usage: {track_err}")
        
        return {
            "response": response_text or "No response generated",
            "has_context": True,
            "experiments_loaded": experiments_context != "No experiments have been run yet.",
            "tools_used": round_num > 0 if 'round_num' in locals() else False
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


def rerun_experiment_command(experiment_id):
    """Rerun a specific experiment with its original code if available."""
    try:
        # Check if we have the original experiment's code
        reasoning_table = boto3.resource('dynamodb', region_name='us-east-1').Table('ai-video-codec-reasoning')
        
        try:
            reasoning_response = reasoning_table.query(
                KeyConditionExpression='reasoning_id = :id',
                ExpressionAttributeValues={':id': experiment_id}
            )
            has_code = False
            if reasoning_response.get('Items'):
                reasoning_item = reasoning_response['Items'][0]
                generated_code = reasoning_item.get('generated_code')
                if generated_code and generated_code != '{}':
                    has_code = True
                    logger.info(f"Found original code for {experiment_id}")
        except Exception as e:
            logger.warning(f"Could not check for original code: {e}")
            has_code = False
        
        # Send SSM command to trigger rerun with experiment ID
        response = ssm.send_command(
            InstanceIds=[ORCHESTRATOR_INSTANCE_ID],
            DocumentName='AWS-RunShellScript',
            Parameters={
                'commands': [
                    'cd /home/ec2-user/ai-video-codec',
                    f'export RERUN_EXPERIMENT_ID={experiment_id}',
                    'python3 src/agents/procedural_experiment_runner.py'
                ]
            },
            TimeoutSeconds=1800,
            Comment=f'Rerun experiment {experiment_id}'
        )
        
        message = f"Experiment rerun started. Command ID: {response['Command']['CommandId']}"
        if not has_code:
            message += " (Warning: Original code not found - will generate new code)"
        
        return {
            "success": True,
            "message": message,
            "has_original_code": has_code
        }
    
    except Exception as e:
        logger.error(f"Error rerunning experiment: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "message": f"Error: {str(e)}", "error_type": type(e).__name__}


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


def set_auto_execution_command(enabled):
    """Enable or disable automatic experiment execution."""
    try:
        value = "true" if enabled else "false"
        ssm.put_parameter(
            Name='/ai-video-codec/auto-execution-enabled',
            Value=value,
            Type='String',
            Overwrite=True
        )
        status = "enabled" if enabled else "disabled"
        logger.info(f"Auto-execution {status}")
        return {
            "success": True,
            "message": f"Auto-execution {status}",
            "enabled": enabled
        }
    except Exception as e:
        logger.error(f"Error setting auto-execution: {e}")
        return {"success": False, "message": str(e)}


def get_auto_execution_status():
    """Get the current auto-execution status."""
    try:
        response = ssm.get_parameter(Name='/ai-video-codec/auto-execution-enabled')
        enabled = response['Parameter']['Value'] == 'true'
        return {
            "success": True,
            "enabled": enabled,
            "last_modified": response['Parameter']['LastModifiedDate'].isoformat()
        }
    except ssm.exceptions.ParameterNotFound:
        return {"success": True, "enabled": False, "message": "Parameter not found - defaulting to disabled"}
    except Exception as e:
        logger.error(f"Error getting auto-execution status: {e}")
        return {"success": False, "message": str(e)}


def purge_experiment_command(experiment_id):
    """Delete a specific experiment from all tables."""
    try:
        deleted_counts = {}
        
        # Delete from experiments table
        try:
            experiments_table.delete_item(
                Key={'experiment_id': experiment_id}
            )
            deleted_counts['experiments'] = 1
            logger.info(f"Deleted experiment {experiment_id} from experiments table")
        except Exception as e:
            logger.warning(f"Error deleting from experiments table: {e}")
            deleted_counts['experiments'] = 0
        
        # Delete from reasoning table
        try:
            reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
            reasoning_table.delete_item(
                Key={'reasoning_id': experiment_id}
            )
            deleted_counts['reasoning'] = 1
            logger.info(f"Deleted reasoning for {experiment_id}")
        except Exception as e:
            logger.warning(f"Error deleting from reasoning table: {e}")
            deleted_counts['reasoning'] = 0
        
        # Delete from metrics table (may have multiple entries)
        try:
            metrics_table = dynamodb.Table('ai-video-codec-metrics')
            # Query for all metrics with this experiment_id
            response = metrics_table.query(
                KeyConditionExpression='experiment_id = :id',
                ExpressionAttributeValues={':id': experiment_id}
            )
            items = response.get('Items', [])
            deleted_count = 0
            with metrics_table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(Key={
                        'experiment_id': item['experiment_id'],
                        'timestamp': item['timestamp']
                    })
                    deleted_count += 1
            deleted_counts['metrics'] = deleted_count
            logger.info(f"Deleted {deleted_count} metrics for {experiment_id}")
        except Exception as e:
            logger.warning(f"Error deleting from metrics table: {e}")
            deleted_counts['metrics'] = 0
        
        total_deleted = sum(deleted_counts.values())
        return {
            "success": True,
            "message": f"Purged experiment {experiment_id} ({total_deleted} records deleted)",
            "deleted_counts": deleted_counts
        }
    
    except Exception as e:
        logger.error(f"Error purging experiment {experiment_id}: {e}")
        return {"success": False, "message": str(e)}


def purge_all_experiments_command():
    """Delete all experiments from all tables."""
    try:
        tables_to_purge = [
            ('ai-video-codec-experiments', ['experiment_id']),
            ('ai-video-codec-reasoning', ['reasoning_id']),
            ('ai-video-codec-metrics', ['experiment_id', 'timestamp'])
        ]
        
        total_deleted = 0
        deleted_by_table = {}
        
        for table_name, key_names in tables_to_purge:
            try:
                table = dynamodb.Table(table_name)
                
                # Scan all items
                response = table.scan()
                items = response.get('Items', [])
                
                if not items:
                    deleted_by_table[table_name] = 0
                    continue
                
                # Batch delete
                deleted = 0
                with table.batch_writer() as batch:
                    for item in items:
                        key = {k: item[k] for k in key_names if k in item}
                        batch.delete_item(Key=key)
                        deleted += 1
                
                deleted_by_table[table_name] = deleted
                total_deleted += deleted
                logger.info(f"Purged {deleted} items from {table_name}")
                
            except Exception as e:
                logger.error(f"Error purging {table_name}: {e}")
                deleted_by_table[table_name] = 0
        
        return {
            "success": True,
            "message": f"Purged all experiments ({total_deleted} total records deleted)",
            "deleted_by_table": deleted_by_table,
            "total_deleted": total_deleted
        }
    
    except Exception as e:
        logger.error(f"Error purging all experiments: {e}")
        return {"success": False, "message": str(e)}


def get_experiments_list():
    """Get list of recent experiments with details."""
    try:
        # Get all experiments and sort to get the latest ones
        response = experiments_table.scan()
        all_items = response.get('Items', [])
        
        # Handle pagination if there are more items
        while 'LastEvaluatedKey' in response:
            response = experiments_table.scan(
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            all_items.extend(response.get('Items', []))
        
        # Sort by timestamp (descending) and take the latest 50
        all_items.sort(key=lambda x: int(x.get('timestamp', 0)), reverse=True)
        total_count = len(all_items)  # Store total count before limiting
        experiments = all_items[:50]  # Get latest 50 experiments
        
        # Also check SSM for running commands
        ssm_commands = []
        try:
            cmd_response = ssm.list_commands(
                InstanceId=ORCHESTRATOR_INSTANCE_ID,
                MaxResults=10
            )
            ssm_commands = cmd_response.get('Commands', [])
        except Exception as e:
            logger.error(f"Error fetching SSM commands: {e}")
        
        # Format experiments for display (already sorted by timestamp descending)
        exp_list = []
        for exp in experiments:
            exp_data = {
                'id': exp.get('experiment_id'),
                'timestamp': exp.get('timestamp'),
                'status': exp.get('status', 'unknown'),
                'duration': 0,
                'best_bitrate': None,
                'psnr_db': None,
                'ssim': None,
                'quality': None,
                'video_url': None,
                'decoder_s3_key': None,
                'experiments_run': 0,
                # Phase tracking
                'current_phase': exp.get('current_phase', 'unknown'),
                'phase_completed': exp.get('phase_completed', 'unknown'),
                'validation_retries': exp.get('validation_retries', 0),
                'execution_retries': exp.get('execution_retries', 0),
                # Runtime tracking
                'start_time': exp.get('start_time'),
                'elapsed_seconds': exp.get('elapsed_seconds', 0),
                'estimated_duration_seconds': exp.get('estimated_duration_seconds', 0),
                # Code evolution fields
                'code_changed': False,
                'version': 0,
                'evolution_status': 'N/A',
                'improvement': 'N/A',
                'summary': '',
                'deployment_status': 'not_deployed',
                'github_committed': False,
                'github_commit_hash': None,
                # Human intervention tracking
                'needs_human': exp.get('needs_human', False),
                'human_intervention_reasons': exp.get('human_intervention_reasons', [])
            }
            
            # Parse experiment details - handle BOTH old and new formats
            try:
                # Check for NEW format (result field)
                result_data = exp.get('result')
                if result_data:
                    # NEW format from GPU workers
                    result = json.loads(result_data) if isinstance(result_data, str) else result_data
                    metrics = result.get('metrics', {})
                    
                    exp_data['best_bitrate'] = float(metrics.get('bitrate_mbps', 0)) if metrics.get('bitrate_mbps') else None
                    exp_data['psnr_db'] = float(metrics.get('psnr_db', 0)) if metrics.get('psnr_db') else None
                    exp_data['ssim'] = float(metrics.get('ssim', 0)) if metrics.get('ssim') else None
                    exp_data['experiments_run'] = 1  # New format = 1 experiment
                    
                    # Calculate quality
                    psnr = metrics.get('psnr_db', 0)
                    if psnr > 35:
                        exp_data['quality'] = 'excellent'
                    elif psnr > 30:
                        exp_data['quality'] = 'good'
                    elif psnr > 25:
                        exp_data['quality'] = 'acceptable'
                    elif psnr > 0:
                        exp_data['quality'] = 'poor'
                    
                    # Get processing time
                    processing_time = metrics.get('processing_time_seconds', 0)
                    exp_data['elapsed_seconds'] = processing_time
                    
                    # Worker info
                    worker_id = result.get('worker_id', '')
                    is_gpu = 'ip-10-0-2-118' in worker_id
                    exp_data['worker'] = 'GPU-Worker-1' if is_gpu else 'CPU'
                    
                else:
                    # OLD format from orchestrator
                    exp_details = exp.get('experiments', '[]')
                    if isinstance(exp_details, str):
                        exp_details = json.loads(exp_details)
                    
                    exp_data['experiments_run'] = len(exp_details)
                    
                    # Extract code evolution data and best bitrate
                    for e in exp_details:
                        # Check for code evolution experiment
                        if e.get('experiment_type') == 'llm_generated_code_evolution':
                            exp_data['code_changed'] = True
                            
                            # Get evolution info
                            evolution = e.get('evolution', {})
                            code_info = e.get('code_info', {})
                            
                            exp_data['version'] = code_info.get('version', 0)
                            exp_data['evolution_status'] = evolution.get('status', 'unknown')
                            exp_data['improvement'] = evolution.get('improvement', 'N/A')
                            exp_data['summary'] = evolution.get('summary', evolution.get('reason', ''))
                            exp_data['deployment_status'] = evolution.get('deployment_status', 'not_deployed')
                            exp_data['github_committed'] = evolution.get('github_committed', False)
                            exp_data['github_commit_hash'] = evolution.get('github_commit_hash', None)
                            
                            # Extract failure analysis if present
                            failure_analysis = evolution.get('failure_analysis', {})
                            if failure_analysis:
                                exp_data['failure_analysis'] = {
                                    'category': failure_analysis.get('failure_category', 'unknown'),
                                    'root_cause': failure_analysis.get('root_cause', 'N/A'),
                                    'fix_suggestion': failure_analysis.get('fix_suggestion', 'N/A'),
                                    'severity': failure_analysis.get('severity', 'unknown')
                                }
                        
                        # Find best bitrate and quality metrics
                        metrics = e.get('real_metrics', {})
                        bitrate = metrics.get('bitrate_mbps')
                        if bitrate:
                            if exp_data['best_bitrate'] is None or bitrate < exp_data['best_bitrate']:
                                exp_data['best_bitrate'] = float(bitrate)
                                # Also extract quality metrics from the best performing experiment
                                exp_data['psnr_db'] = float(metrics.get('psnr_db')) if metrics.get('psnr_db') else None
                                exp_data['ssim'] = float(metrics.get('ssim')) if metrics.get('ssim') else None
                                exp_data['quality'] = metrics.get('quality')
                                # Extract media artifacts (video and decoder code)
                                exp_data['video_url'] = e.get('video_url')
                                exp_data['decoder_s3_key'] = e.get('decoder_s3_key')
            except Exception as e:
                logger.error(f"Error parsing experiment {exp_data['id']}: {e}")
            
            exp_list.append(exp_data)
        
        # Check for any currently running commands
        running_commands = [cmd for cmd in ssm_commands if cmd['Status'] == 'InProgress']
        
        return {
            "success": True,
            "total_count": total_count,
            "experiments": exp_list,
            "running_commands": len(running_commands)
        }
    
    except Exception as e:
        logger.error(f"Error getting experiments list: {e}")
        return {"success": False, "error": str(e), "experiments": []}


def handle_command(command, params=None):
    """Execute admin command."""
    command_map = {
        'start_experiment': start_experiment_command,
        'stop_experiments': stop_experiments_command,
        'pause_autonomous': pause_autonomous_command,
        'resume_autonomous': resume_autonomous_command,
        'rerun_experiment': rerun_experiment_command,
        'set_auto_execution': set_auto_execution_command,
        'purge_experiment': purge_experiment_command,
        'purge_all_experiments': purge_all_experiments_command,
    }
    
    handler = command_map.get(command.lower())
    if handler:
        # Check if command requires parameters
        if command.lower() == 'rerun_experiment':
            if params and 'experiment_id' in params:
                return handler(params['experiment_id'])
            else:
                return {"success": False, "message": "experiment_id required for rerun_experiment"}
        elif command.lower() == 'set_auto_execution':
            if params and 'enabled' in params:
                return handler(params['enabled'])
            else:
                return {"success": False, "message": "enabled (true/false) required for set_auto_execution"}
        elif command.lower() == 'purge_experiment':
            if params and 'experiment_id' in params:
                return handler(params['experiment_id'])
            else:
                return {"success": False, "message": "experiment_id required for purge_experiment"}
        else:
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
            # Check for Authorization header (case-insensitive for API Gateway compatibility)
            headers_dict = event.get('headers', {})
            auth_header = headers_dict.get('Authorization', '') or headers_dict.get('authorization', '')
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
        
        elif path == '/admin/experiments':
            response_body = get_experiments_list()
        
        elif path == '/admin/chat' and method == 'POST':
            message = body.get('message', '')
            # Get chat history from DynamoDB if not provided
            history = body.get('history', [])
            if not history:
                try:
                    history_response = control_table.get_item(Key={'control_id': 'chat_history'})
                    history_item = history_response.get('Item', {})
                    history = json.loads(history_item.get('messages', '[]'))
                except Exception as e:
                    logger.error(f"Failed to load chat history: {e}")
                    history = []
            
            # Call LLM
            llm_response = call_llm_chat(message, history)
            
            # Store chat message in history only if LLM call succeeded
            if 'response' in llm_response and 'error' not in llm_response:
                try:
                    history.append({'role': 'user', 'content': message, 'timestamp': datetime.utcnow().isoformat()})
                    history.append({'role': 'assistant', 'content': llm_response['response'], 'timestamp': datetime.utcnow().isoformat()})
                    
                    # Keep only last 50 messages
                    history = history[-50:]
                    
                    control_table.put_item(Item={
                        'control_id': 'chat_history',
                        'messages': json.dumps(history, default=str),
                        'updated_at': datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Failed to store chat history: {e}")
            
            response_body = llm_response
        
        elif path == '/admin/chat' and method == 'GET':
            # Return chat history
            try:
                chat_history_response = control_table.get_item(Key={'control_id': 'chat_history'})
                chat_history_item = chat_history_response.get('Item', {})
                messages = json.loads(chat_history_item.get('messages', '[]'))
                response_body = {'messages': messages}
            except Exception as e:
                logger.error(f"Failed to load chat history: {e}")
                response_body = {'messages': []}
        
        elif path == '/admin/command' and method == 'POST':
            command = body.get('command', '')
            response_body = handle_command(command, body)
        
        elif path == '/admin/auto-execution' and method == 'GET':
            response_body = get_auto_execution_status()
        
        elif path == '/admin/execute' and method == 'POST':
            command = body.get('command', '')
            response_body = handle_command(command, body)
        
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

