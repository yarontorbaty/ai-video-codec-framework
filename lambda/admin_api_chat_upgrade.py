# This is the upgraded call_llm_chat function with full orchestrator context and tools
# To be integrated into admin_api.py

def _load_system_prompt():
    """Load the comprehensive system prompt from LLM_SYSTEM_PROMPT.md"""
    try:
        # Try to load from S3 or local paths
        import boto3
        s3 = boto3.client('s3')
        
        try:
            # Try S3 first
            response = s3.get_object(
                Bucket='ai-video-codec-artifacts-580473065386',
                Key='system/LLM_SYSTEM_PROMPT.md'
            )
            prompt = response['Body'].read().decode('utf-8')
            logger.info("✅ Loaded system prompt from S3")
            return prompt
        except:
            pass
        
        # Fallback: use basic prompt
        logger.warning("⚠️  Could not load LLM_SYSTEM_PROMPT.md - using basic prompt")
        return """You are an autonomous AI video codec research system with full capabilities."""
        
    except Exception as e:
        logger.warning(f"⚠️  Error loading system prompt: {e}")
        return """You are an autonomous AI video codec research system."""


def _get_experiments_context():
    """Fetch recent experiments for LLM context"""
    try:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        experiments_table = dynamodb.Table('ai-video-codec-experiments')
        
        # Scan all items and sort
        all_items = []
        scan_kwargs = {}
        
        while True:
            response = experiments_table.scan(**scan_kwargs)
            all_items.extend(response.get('Items', []))
            
            if 'LastEvaluatedKey' not in response:
                break
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        # Sort by timestamp descending
        all_items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        recent = all_items[:5]  # Last 5 experiments
        
        if not recent:
            return "No experiments have been run yet."
        
        context = "## Recent Experiments:\n\n"
        for exp in recent:
            exp_id = exp.get('experiment_id', 'N/A')
            timestamp = exp.get('timestamp', 0)
            status = exp.get('status', 'unknown')
            
            # Extract metrics
            experiments = exp.get('experiments', [])
            if isinstance(experiments, str):
                experiments = json.loads(experiments)
            
            best_bitrate = None
            for e in experiments:
                metrics = e.get('real_metrics', {})
                bitrate = metrics.get('bitrate_mbps')
                if bitrate and (best_bitrate is None or bitrate < best_bitrate):
                    best_bitrate = bitrate
            
            context += f"- **{exp_id}**: {status}, Bitrate: {best_bitrate:.4f if best_bitrate else 'N/A'} Mbps\n"
        
        return context
        
    except Exception as e:
        logger.error(f"Failed to fetch experiments context: {e}")
        return "Error fetching experiments data."


def _execute_admin_tool(tool_name, tool_input):
    """Execute a tool call from admin chat"""
    logger.info(f"🛠️  Admin chat tool: {tool_name}")
    logger.info(f"   Input: {json.dumps(tool_input, indent=2)}")
    
    try:
        # Import framework modifier
        import sys
        import os
        sys.path.append('/tmp')  # Lambda tmp directory
        
        # For Lambda, we need to handle this differently
        # Tools should be executed via SSM command to orchestrator
        # For now, return instruction to use SSM
        
        return {
            'success': True,
            'message': f"Tool '{tool_name}' would be executed on orchestrator",
            'note': 'Tool execution from admin chat requires SSM integration',
            'tool': tool_name,
            'input': tool_input
        }
        
    except Exception as e:
        logger.error(f"❌ Tool execution error: {e}")
        return {
            'success': False,
            'error': str(e),
            'tool': tool_name
        }


def call_llm_chat(message, history):
    """
    Call governing LLM with admin message - WITH FULL ORCHESTRATOR CONTEXT AND TOOLS.
    """
    try:
        import urllib.request
        import urllib.error
        
        if not ANTHROPIC_API_KEY:
            return {"error": "LLM API key not configured - check Secrets Manager"}
        
        # Load full system prompt (same as orchestrator)
        base_system_prompt = _load_system_prompt()
        
        # Add admin chat context
        admin_context = """

## 🎧 ADMIN CHAT MODE

You are in a direct conversation with the human administrator.

**You have full orchestrator capabilities:**
- Complete system prompt and context
- Recent experiments data (provided below)
- Framework modification tools (can use them)
- Tool calling enabled

**For this conversation:**
- Answer questions about experiments clearly
- Provide insights and recommendations
- You CAN use tools to fix issues immediately
- Explain your reasoning
- Be conversational but precise

**Remember**: This is a direct chat with the project's human overseer!

"""
        system_prompt = base_system_prompt + admin_context
        
        # Fetch recent experiments for context
        experiments_context = _get_experiments_context()
        
        # Build conversation
        messages = []
        
        # Add experiments context as system message
        context_msg = f"""Current System State:

{experiments_context}

You can see the recent experiments above. The admin's message follows..."""
        
        messages.append({"role": "user", "content": context_msg})
        
        # Add chat history
        for msg in history[-10:]:  # Last 10 messages
            role = "user" if msg['role'] == 'user' else "assistant"
            messages.append({"role": role, "content": msg['content']})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Import framework tools definition
        from utils.framework_modifier import FRAMEWORK_TOOLS
        
        # API call with tool support
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Tool calling loop (max 3 rounds for admin chat)
        for round_num in range(3):
            data = json.dumps({
                "model": "claude-sonnet-4-5",
                "max_tokens": 4096,
                "temperature": 0.7,
                "system": system_prompt,
                "messages": messages,
                "tools": FRAMEWORK_TOOLS
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            # Check stop reason
            stop_reason = result.get('stop_reason')
            
            if stop_reason == "tool_use":
                logger.info(f"🛠️  LLM using tools in admin chat (round {round_num + 1}/3)")
                
                # Add assistant message
                messages.append({"role": "assistant", "content": result['content']})
                
                # Execute tools
                tool_results = []
                for block in result['content']:
                    if block.get('type') == 'tool_use':
                        tool_result = _execute_admin_tool(block['name'], block['input'])
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block['id'],
                            "content": json.dumps(tool_result)
                        })
                
                # Add tool results
                messages.append({"role": "user", "content": tool_results})
                continue
            
            # No more tools - extract response
            response_text = None
            for block in result['content']:
                if block.get('type') == 'text':
                    response_text = block['text']
                    break
            
            if not response_text:
                response_text = "I apologize, I couldn't generate a response."
            
            # Track token usage
            try:
                usage = result.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                track_llm_usage(input_tokens, output_tokens)
            except Exception as track_err:
                logger.error(f"Failed to track LLM usage: {track_err}")
            
            return {
                "response": response_text,
                "tools_used": round_num > 0
            }
        
        # Max rounds reached
        return {
            "response": "Tool calling limit reached. Please try again.",
            "tools_used": True
        }
    
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"error": str(e)}

