#!/usr/bin/env python3
"""
LLM-Powered Experiment Planner
Uses Claude/GPT to analyze experiment results and plan next experiments autonomously.
"""

import os
import json
import boto3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("anthropic library not available - will try direct API calls")

# Import framework modifier for tool calling
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.framework_modifier import FrameworkModifier, FRAMEWORK_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMExperimentPlanner:
    """
    Autonomous experiment planner using LLM reasoning.
    Analyzes past experiments and plans improved approaches.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5"):
        """
        Initialize LLM planner.
        
        Args:
            model: LLM model to use (default: Claude Sonnet 4.5 - latest model)
        """
        self.model = model
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.s3 = boto3.client('s3', region_name='us-east-1')
        
        # Initialize Anthropic client (if library and API key available)
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("LLM planning enabled with Claude API")
        else:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("anthropic library not installed - using fallback planning")
            elif not api_key:
                logger.warning("No ANTHROPIC_API_KEY found - using fallback planning")
            self.client = None
        
        self.experiments_table = self.dynamodb.Table('ai-video-codec-experiments')
        self.reasoning_table = self.dynamodb.Table('ai-video-codec-reasoning')
        
        # Enable direct API calls as fallback
        self.api_key = api_key
        self.use_direct_api = (not ANTHROPIC_AVAILABLE and api_key)
        
        # Load system prompt from file
        self.system_prompt = self._load_system_prompt()
        
        # Initialize framework modifier for tool calling
        self.framework_modifier = FrameworkModifier()
        self.tool_calling_enabled = True  # Enable meta-level autonomy
    
    def _load_system_prompt(self) -> str:
        """Load the comprehensive system prompt from LLM_SYSTEM_PROMPT.md"""
        try:
            # Try multiple possible paths
            possible_paths = [
                'LLM_SYSTEM_PROMPT.md',  # From project root
                '../LLM_SYSTEM_PROMPT.md',  # From src/agents/
                '../../LLM_SYSTEM_PROMPT.md',  # From deeper
                '/home/ec2-user/ai-video-codec/LLM_SYSTEM_PROMPT.md',  # Absolute on EC2
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        prompt = f.read()
                    logger.info(f"✅ Loaded system prompt from {path} ({len(prompt)} chars)")
                    return prompt
            
            logger.warning("⚠️  Could not find LLM_SYSTEM_PROMPT.md - using basic prompt")
            return ""
            
        except Exception as e:
            logger.warning(f"⚠️  Error loading system prompt: {e}")
            return ""
    
    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """
        Execute a tool call from the LLM.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        logger.info(f"🛠️  LLM requested tool: {tool_name}")
        logger.info(f"   Input: {json.dumps(tool_input, indent=2)}")
        
        try:
            if tool_name == "modify_framework_file":
                result = self.framework_modifier.modify_file(
                    file_path=tool_input['file_path'],
                    modification_type=tool_input['modification_type'],
                    content=tool_input['content'],
                    reason=tool_input['reason']
                )
                
            elif tool_name == "run_shell_command":
                result = self.framework_modifier.run_command(
                    command=tool_input['command'],
                    reason=tool_input['reason']
                )
                
            elif tool_name == "install_python_package":
                result = self.framework_modifier.install_package(
                    package=tool_input['package'],
                    reason=tool_input['reason']
                )
                
            elif tool_name == "restart_orchestrator":
                result = self.framework_modifier.restart_orchestrator()
                
            elif tool_name == "rollback_file":
                result = self.framework_modifier.rollback_file(
                    file_path=tool_input['file_path']
                )
                
            else:
                result = {
                    'success': False,
                    'error': f"Unknown tool: {tool_name}"
                }
            
            logger.info(f"✅ Tool result: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'tool': tool_name
            }
    
    def _call_claude_direct(self, prompt: str) -> Optional[str]:
        """
        Call Claude API directly using requests (Python 3.7 compatible).
        Fallback when anthropic library not available.
        """
        if not self.api_key:
            return None
        
        try:
            import urllib.request
            import urllib.error
            
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = json.dumps({
                "model": self.model,
                "max_tokens": 4096,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}]
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['content'][0]['text']
                
        except Exception as e:
            logger.error(f"Direct API call failed: {e}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response with new sectioned format.
        Handles ANALYSIS (JSON) and NEURAL CODEC IMPLEMENTATION (Python code) sections.
        """
        try:
            result = {}
            
            # Extract JSON from ANALYSIS section
            if "## ANALYSIS" in response_text and "```json" in response_text:
                logger.info("   Detected new sectioned format")
                
                # Extract JSON section
                analysis_start = response_text.find("## ANALYSIS")
                json_start = response_text.find("```json", analysis_start)
                if json_start != -1:
                    json_end = response_text.find("```", json_start + 7)
                    if json_end != -1:
                        json_str = response_text[json_start + 7:json_end].strip()
                        try:
                            analysis = json.loads(json_str)
                            result.update(analysis)
                            logger.info("   ✅ Successfully parsed ANALYSIS JSON")
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse failed: {e}")
                            return self._extract_fields_manually(response_text)
            
            # Extract Python code from NEURAL CODEC IMPLEMENTATION section
            if "## NEURAL CODEC IMPLEMENTATION" in response_text and "```python" in response_text:
                # Extract Python code section
                code_start_marker = response_text.find("## NEURAL CODEC IMPLEMENTATION")
                python_start = response_text.find("```python", code_start_marker)
                if python_start != -1:
                    python_end = response_text.find("```", python_start + 9)
                    if python_end != -1:
                        code = response_text[python_start + 9:python_end].strip()
                        result['encoding_agent_code'] = code
                        result['decoding_agent_code'] = code
                        logger.info(f"   ✅ Extracted Python code ({len(code)} chars)")
            
            # Fallback to old format if new format not detected
            if not result:
                logger.info("   No new format detected, trying old format...")
                return self._extract_fields_manually(response_text)
            
            return result if result else None
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
    
    def _extract_fields_manually(self, text: str) -> Optional[Dict]:
        """
        Manually extract fields from LLM response when JSON parsing fails.
        Looks for code blocks and key fields.
        """
        import re
        
        result = {}
        
        try:
            # Check for v1 format with "generated_code"
            if '"generated_code"' in text:
                logger.info("   Detected v1 format with generated_code field")
                
                # Try multiple patterns for v1 format
                patterns = [
                    # Pattern 1: Standard JSON with escaped quotes
                    r'"generated_code"\s*:\s*\{[^}]*"code"\s*:\s*"([^"]*(?:(?:\\"|\\\n)[^"]*)*)"',
                    # Pattern 2: Multi-line string with triple quotes
                    r'"generated_code"\s*:\s*\{[^}]*"code"\s*:\s*"""(.*?)"""',
                    # Pattern 3: Raw string without escaping (but this will break on quotes)
                    r'"generated_code"\s*:\s*\{[^}]*"code"\s*:\s*"([^"]*)"',
                ]
                
                code = None
                for pattern in patterns:
                    code_match = re.search(pattern, text, re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                        if pattern == patterns[0]:  # Escaped quotes pattern
                            code = code.replace('\\"', '"').replace('\\n', '\n')
                        break
                
                # If standard patterns fail, try to extract the full code block
                # by finding the start of the code and extracting until the next field
                if not code or len(code) < 100:
                    logger.info("   🔍 Trying to extract full code block from malformed JSON...")
                    
                    # Find the start of the code field
                    code_start = re.search(r'"generated_code"\s*:\s*\{[^}]*"code"\s*:\s*"', text)
                    if code_start:
                        start_pos = code_start.end()
                        
                        # Find the end by looking for the closing quote before the next field
                        # Look for patterns like ", "description": or ", "expected_improvement":
                        end_patterns = [
                            r'",\s*"description"\s*:',
                            r'",\s*"expected_improvement"\s*:',
                            r'",\s*"}\s*$',
                            r'"\s*}\s*$'
                        ]
                        
                        for end_pattern in end_patterns:
                            end_match = re.search(end_pattern, text[start_pos:])
                            if end_match:
                                code = text[start_pos:start_pos + end_match.start()]
                                # Unescape the code
                                code = code.replace('\\"', '"').replace('\\n', '\n')
                                logger.info(f"   ✅ Extracted full code block ({len(code)} chars)")
                                break
                        
                        # If still no code found, try a more aggressive approach
                        if not code or len(code) < 100:
                            logger.info("   🔍 Trying aggressive code extraction...")
                            
                            # Look for the actual code content by finding the start of Python code
                            python_start = re.search(r'import numpy as np', text[start_pos:])
                            if python_start:
                                python_start_pos = start_pos + python_start.start()
                                
                                # Find the end by looking for the next JSON field
                                python_end_patterns = [
                                    r'",\s*"description"\s*:',
                                    r'",\s*"expected_improvement"\s*:',
                                    r'"\s*}\s*$'
                                ]
                                
                                for end_pattern in python_end_patterns:
                                    end_match = re.search(end_pattern, text[python_start_pos:])
                                    if end_match:
                                        code = text[python_start_pos:python_start_pos + end_match.start()]
                                        # Unescape the code
                                        code = code.replace('\\"', '"').replace('\\n', '\n')
                                        logger.info(f"   ✅ Extracted Python code block ({len(code)} chars)")
                                        break
                
                if code:
                    result['encoding_agent_code'] = code
                    result['decoding_agent_code'] = code
                    logger.info(f"   ✅ Extracted v1 code ({len(code)} chars)")
                else:
                    logger.warning("   ⚠️  Could not extract v1 code with any pattern")
            
            # Try v2 format
            else:
                # Extract encoding_agent_code with multiple patterns
                encoding_patterns = [
                    r'"encoding_agent_code"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'"encoding_agent_code"\s*:\s*"""(.*?)"""',
                    r'```python\s*#\s*ENCODING\s*AGENT\s*(.*?)```',
                    r'```python.*?def run_encoding_agent.*?(.*?)```',
                ]
                
                encoding_code = None
                for pattern in encoding_patterns:
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        encoding_code = (match.group(1) or "").strip()
                        if len(encoding_code) > 50:  # Reasonable minimum for real code
                            break
                
                if encoding_code:
                    result['encoding_agent_code'] = encoding_code
                
                # Extract decoding_agent_code with multiple patterns
                decoding_patterns = [
                    r'"decoding_agent_code"\s*:\s*"((?:[^"\\]|\\.)*)"',
                    r'"decoding_agent_code"\s*:\s*"""(.*?)"""',
                    r'```python\s*#\s*DECODING\s*AGENT\s*(.*?)```',
                    r'```python.*?def run_decoding_agent.*?(.*?)```',
                ]
                
                decoding_code = None
                for pattern in decoding_patterns:
                    match = re.search(pattern, text, re.DOTALL)
                    if match:
                        decoding_code = (match.group(1) or "").strip()
                        if len(decoding_code) > 50:  # Reasonable minimum for real code
                            break
                
                if decoding_code:
                    result['decoding_agent_code'] = decoding_code
                
                # If we still don't have code, try to extract any Python code blocks
                if not result.get('encoding_agent_code') or len(result.get('encoding_agent_code', '')) < 50:
                    logger.info("   🔍 Trying to extract any Python code blocks...")
                    
                    # Look for any Python code blocks
                    python_blocks = re.findall(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
                    if python_blocks:
                        # Use the longest code block (likely the main implementation)
                        longest_block = max(python_blocks, key=len)
                        if len(longest_block) > 50:
                            result['encoding_agent_code'] = longest_block.strip()
                            result['decoding_agent_code'] = longest_block.strip()
                            logger.info(f"   ✅ Extracted Python code block ({len(longest_block)} chars)")
                
                # Final fallback: look for function definitions
                if not result.get('encoding_agent_code') or len(result.get('encoding_agent_code', '')) < 50:
                    logger.info("   🔍 Trying to extract function definitions...")
                    
                    # Look for function definitions
                    func_match = re.search(r'(def\s+\w+.*?)(?=\ndef|\Z)', text, re.DOTALL)
                    if func_match:
                        func_code = func_match.group(1).strip()
                        if len(func_code) > 50:
                            result['encoding_agent_code'] = func_code
                            result['decoding_agent_code'] = func_code
                            logger.info(f"   ✅ Extracted function code ({len(func_code)} chars)")
            
            # Extract other fields
            for field in ['hypothesis', 'compression_strategy', 'expected_bitrate_mbps']:
                match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', text)
                if match:
                    result[field] = match.group(1)
                else:
                    match = re.search(rf'"{field}"\s*:\s*([0-9.]+)', text)
                    if match:
                        result[field] = float(match.group(1))
            
            # If we got both agent codes, consider it a success
            if 'encoding_agent_code' in result and 'decoding_agent_code' in result:
                logger.info("✅ Manually extracted agent codes from malformed JSON")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Manual extraction failed: {e}")
            return None
    
    def analyze_recent_experiments(self, limit: int = 5) -> List[Dict]:
        """
        Fetch and analyze recent experiments from DynamoDB.
        
        Args:
            limit: Number of recent experiments to analyze
            
        Returns:
            List of experiment data dictionaries
        """
        logger.info(f"Fetching {limit} most recent experiments...")
        
        try:
            response = self.experiments_table.scan(Limit=limit)
            experiments = response.get('Items', [])
            
            # Sort by timestamp (most recent first)
            experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            logger.info(f"Retrieved {len(experiments)} experiments")
            return experiments
            
        except Exception as e:
            logger.error(f"Error fetching experiments: {e}")
            return []
    
    def generate_analysis_prompt(self, experiments: List[Dict]) -> str:
        """
        Generate detailed prompt for LLM analysis.
        
        Args:
            experiments: List of experiment data
            
        Returns:
            Formatted prompt string
        """
        prompt = """You are an expert AI researcher working on a novel video codec that uses neural networks and procedural generation to achieve 90% bitrate reduction vs HEVC while maintaining PSNR > 95%.

# CURRENT GOAL
Beat 10 Mbps HEVC baseline with < 1 Mbps (90% reduction) while maintaining quality.

# RECENT EXPERIMENT RESULTS
"""
        
        for i, exp in enumerate(experiments, 1):
            exp_id = exp.get('experiment_id', 'unknown')
            timestamp = exp.get('timestamp_iso', 'unknown')
            status = exp.get('status', 'unknown')
            
            # Parse experiments JSON
            experiments_data = json.loads(exp.get('experiments', '[]'))
            
            prompt += f"\n## Experiment {i}: {exp_id}\n"
            prompt += f"**Time:** {timestamp}\n"
            prompt += f"**Status:** {status}\n\n"
            
            for exp_data in experiments_data:
                exp_type = exp_data.get('experiment_type', 'unknown')
                exp_status = exp_data.get('status', 'unknown')
                
                if exp_type == 'real_procedural_generation':
                    metrics = exp_data.get('real_metrics', {})
                    comparison = exp_data.get('comparison', {})
                    
                    prompt += f"### Procedural Generation\n"
                    prompt += f"- Status: {exp_status}\n"
                    prompt += f"- Bitrate: {metrics.get('bitrate_mbps', 0):.2f} Mbps\n"
                    prompt += f"- File Size: {metrics.get('file_size_mb', 0):.2f} MB\n"
                    prompt += f"- Resolution: {metrics.get('resolution', 'N/A')}\n"
                    prompt += f"- Duration: {metrics.get('duration', 0)} seconds\n"
                    prompt += f"- HEVC Baseline: {comparison.get('hevc_baseline_mbps', 0)} Mbps\n"
                    prompt += f"- Reduction: {comparison.get('reduction_percent', 0):.1f}%\n"
                    prompt += f"- Target Achieved: {comparison.get('target_achieved', False)}\n\n"
                    
                elif exp_type == 'real_ai_neural_networks':
                    networks = exp_data.get('neural_networks', {})
                    
                    prompt += f"### AI Neural Networks\n"
                    prompt += f"- Status: {exp_status}\n"
                    prompt += f"- Semantic Encoder: {networks.get('semantic_encoder', 'N/A')}\n"
                    prompt += f"- Motion Predictor: {networks.get('motion_predictor', 'N/A')}\n"
                    prompt += f"- Generative Refiner: {networks.get('generative_refiner', 'N/A')}\n"
                    prompt += f"- PyTorch: {networks.get('pytorch_version', 'N/A')}\n\n"
        
        prompt += """
# YOUR TASK

Analyze these results and provide:

1. **Root Cause Analysis**: Why is compression failing? (Be specific and technical)
2. **Key Insights**: What patterns do you see across experiments?
3. **Hypothesis**: What changes would most likely improve results?
4. **Next Experiment Plan**: Concrete steps for the next experiment with:
   - Specific code changes needed
   - Expected improvements
   - Success metrics
5. **Risk Assessment**: What could go wrong with this approach?

# CRITICAL VALIDATION CHECKS

⚠️ **IMPORTANT**: If you see multiple experiments with IDENTICAL metrics (same bitrate, file size, reduction %), this indicates a CODE BUG where experiments are measuring the same file instead of generating unique outputs. In this case:
- Flag this as a critical bug in your root cause analysis
- Recommend checking that output filenames are unique per experiment
- Recommend verifying file paths are using variables, not hardcoded values
- Do NOT proceed with codec improvements until this instrumentation bug is fixed

Be direct, technical, and actionable. Focus on the fundamental problem that procedural generation is creating NEW video (18MB) instead of COMPRESSING existing video.

# CODE GENERATION

**CRITICAL REQUIREMENT**: Generate BOTH compression AND decompression functions!

The codec MUST be able to reconstruct frames for quality verification (PSNR/SSIM).

Required functions:

1. **compress_video_frame()**
   - Signature: `def compress_video_frame(frame: np.ndarray, frame_index: int, config: dict) -> bytes:`
   - Compresses frame to bytes
   
2. **decompress_video_frame()**
   - Signature: `def decompress_video_frame(compressed_data: bytes, frame_index: int, config: dict) -> np.ndarray:`
   - Reconstructs frame from compressed bytes
   - Must return numpy array (H, W, 3) in BGR format
   - Quality target: PSNR >= 30 dB (acceptable) or PSNR >= 35 dB (good)

Requirements:
- Use allowed imports: numpy, cv2, json, struct, base64, math, torch (if needed)
- Both functions must work together (compress → decompress → similar frame)
- Must be runnable in Python 3.7
- Prioritize: bitrate < 5 Mbps AND PSNR >= 30 dB

**Quality vs Bitrate Tradeoff:**
- PSNR < 25 dB: Poor quality (blocky/blurry)
- PSNR 25-30 dB: Acceptable quality
- PSNR 30-35 dB: Good quality (target!)
- PSNR > 35 dB: Excellent quality (H.264/HEVC level)

# OUTPUT FORMAT

**CRITICAL**: Format your response with these EXACT sections:

## ANALYSIS
```json
{
  "root_cause": "Why compression is failing",
  "insights": ["Key patterns observed"],
  "hypothesis": "What changes would improve results",
  "next_experiment": {
    "approach": "Concrete steps for next experiment",
    "changes": ["Specific changes needed"]
  },
  "risks": ["What could go wrong"],
  "expected_bitrate_mbps": 4.2,
  "expected_psnr_db": 32.0,
  "confidence_score": 0.75
}
```

## NEURAL CODEC IMPLEMENTATION
```python
# Your complete neural codec implementation here
# Include BOTH compress_video_frame() AND decompress_video_frame() functions
# Use proper Python syntax - no JSON escaping needed

import numpy as np
import cv2
import struct
# ... your full implementation ...

def compress_video_frame(frame: np.ndarray, frame_index: int, config: dict) -> bytes:
    # Your compression logic here
    pass

def decompress_video_frame(compressed_data: bytes, frame_index: int, config: dict) -> np.ndarray:
    # Your decompression logic here
    pass
```

## DESCRIPTION
Brief description of your neural codec approach and expected performance improvements.

**IMPORTANT**: Do NOT embed code in JSON strings. Use the sectioned format above.
"""
        
        return prompt
    
    def get_llm_analysis(self, experiments: List[Dict]) -> Optional[Dict]:
        """
        Get LLM analysis of experiments.
        
        Args:
            experiments: List of experiment data
            
        Returns:
            Dict with LLM analysis and recommendations, or None if LLM unavailable
        """
        # Try direct API call if anthropic library not available
        if not self.client and self.use_direct_api:
            logger.info("Using direct API call (Python 3.7 compatible)...")
            return self._get_analysis_via_direct_api(experiments)
        
        if not self.client:
            logger.error("LLM client not available - cannot provide analysis")
            return {
                "error": "LLM_NOT_AVAILABLE",
                "message": "Orchestrator LLM not available. Set ANTHROPIC_API_KEY to enable.",
                "root_cause": "N/A - LLM unavailable",
                "insights": ["LLM analysis requires ANTHROPIC_API_KEY environment variable"],
                "hypothesis": "N/A - LLM unavailable",
                "next_experiment": {"approach": "N/A - LLM unavailable"},
                "risks": [],
                "expected_bitrate_mbps": None,
                "confidence_score": 0.0
            }
        
        try:
            prompt = self.generate_analysis_prompt(experiments)
            
            logger.info("Requesting LLM analysis via anthropic library (with tool calling)...")
            
            # Tool calling loop
            messages = [{"role": "user", "content": prompt}]
            
            for round_num in range(5):  # Max 5 tool-use rounds
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.7,
                    messages=messages,
                    tools=FRAMEWORK_TOOLS if self.tool_calling_enabled else None
                )
                
                # Check if LLM wants to use tools
                if message.stop_reason == "tool_use":
                    logger.info(f"🛠️  LLM using tools (round {round_num + 1}/5)")
                    
                    # Add assistant message
                    messages.append({"role": "assistant", "content": message.content})
                    
                    # Execute all tool calls
                    tool_results = []
                    for block in message.content:
                        if block.type == "tool_use":
                            result = self._execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result)
                            })
                    
                    # Add tool results
                    messages.append({"role": "user", "content": tool_results})
                    continue
                
                # No more tools - extract final response
                response_text = None
                for block in message.content:
                    if hasattr(block, 'text'):
                        response_text = block.text
                        break
                
                break
            
            if not response_text:
                logger.warning("No text response after tool calling")
                return None
            
            # Debug: Save response to file for inspection
            try:
                import tempfile
                debug_path = os.path.join(tempfile.gettempdir(), 'llm_response_debug.txt')
                with open(debug_path, 'w') as f:
                    f.write(response_text)
                logger.info(f"   💾 Debug: Response saved to {debug_path}")
            except:
                pass
            
            # Parse response with robust handling
            analysis = self._parse_llm_response(response_text)
            
            if analysis:
                logger.info("✅ LLM analysis received successfully")
            else:
                logger.error("❌ Failed to parse LLM response")
                logger.info(f"   Response preview (first 500 chars): {response_text[:500]}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return None
    
    def _get_analysis_via_direct_api(self, experiments: List[Dict]) -> Optional[Dict]:
        """Get LLM analysis using direct API call."""
        try:
            prompt = self.generate_analysis_prompt(experiments)
            response_text = self._call_claude_direct(prompt)
            
            if not response_text:
                logger.error("Direct API call returned no response")
                return {
                    "error": "LLM_API_FAILED",
                    "message": "LLM API call failed - no response received",
                    "root_cause": "N/A - LLM API error",
                    "insights": ["LLM API did not return a response"],
                    "hypothesis": "N/A - LLM API error",
                    "next_experiment": {"approach": "N/A - LLM API error"},
                    "risks": [],
                    "expected_bitrate_mbps": None,
                    "confidence_score": 0.0
                }
            
            # Parse response with robust handling
            analysis = self._parse_llm_response(response_text)
            
            if analysis:
                logger.info("✅ LLM analysis received via direct API")
            else:
                logger.error("❌ Failed to parse direct API response")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in direct API analysis: {e}")
            return {
                "error": "LLM_API_ERROR",
                "message": f"LLM API error: {str(e)}",
                "root_cause": "N/A - LLM API error",
                "insights": [f"Error calling LLM: {str(e)}"],
                "hypothesis": "N/A - LLM API error",
                "next_experiment": {"approach": "N/A - LLM API error"},
                "risks": [],
                "expected_bitrate_mbps": None,
                "confidence_score": 0.0
            }
    
    def _fallback_analysis(self, experiments: List[Dict]) -> Dict:
        """
        Fallback analysis when LLM is unavailable.
        Uses rule-based logic.
        """
        latest = experiments[0] if experiments else {}
        experiments_data = json.loads(latest.get('experiments', '[]'))
        
        # Extract metrics
        procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
        metrics = procedural.get('real_metrics', {})
        comparison = procedural.get('comparison', {})
        
        bitrate = metrics.get('bitrate_mbps', 15.0)
        reduction = comparison.get('reduction_percent', -50.0)
        
        return {
            "root_cause": "Procedural generation is rendering full video frames (18MB) instead of storing compact procedural parameters (<1KB). The system generates NEW content rather than compressing EXISTING content.",
            "insights": [
                "All experiments show 15 Mbps output (50% LARGER than 10 Mbps HEVC baseline)",
                "Neural networks are operational but not integrated into compression pipeline",
                "The fundamental approach is backwards - we're creating data, not compressing it"
            ],
            "hypothesis": "Store procedural generation PARAMETERS (function types, coefficients, timestamps) instead of rendered frames. Each frame could be described in ~100 bytes instead of ~600KB.",
            "next_experiment": {
                "approach": "Encode video as a sequence of procedural commands",
                "changes": [
                    "Analyze input video to detect procedural patterns",
                    "Store only generation parameters in compact format",
                    "Implement decoder that regenerates frames from parameters",
                    "Measure parameter storage size vs rendered video size"
                ],
                "expected_improvement": "Reduce from 15 Mbps to < 1 Mbps"
            },
            "risks": [
                "Input video may not be procedurally representable",
                "Quality loss if procedural approximation is poor",
                "Decoder complexity may be too high for real-time"
            ],
            "expected_bitrate_mbps": 0.8,
            "confidence_score": 0.75
        }
    
    def log_reasoning(self, analysis: Dict, experiment_id: str) -> None:
        """
        Log LLM reasoning to DynamoDB for dashboard visibility.
        
        Args:
            analysis: LLM analysis results
            experiment_id: ID of the experiment being analyzed
        """
        try:
            from decimal import Decimal
            import uuid
            # Use experiment_id + random UUID to ensure uniqueness
            reasoning_id = f"reasoning_{experiment_id}_{uuid.uuid4().hex[:8]}"
            current_time = int(datetime.utcnow().timestamp())
            
            self.reasoning_table.put_item(Item={
                'reasoning_id': reasoning_id,
                'experiment_id': experiment_id,
                'timestamp': current_time,
                'timestamp_iso': datetime.utcnow().isoformat(),
                'model': self.model,
                'root_cause': analysis.get('root_cause', ''),
                'insights': json.dumps(analysis.get('insights', [])),
                'hypothesis': analysis.get('hypothesis', ''),
                'next_experiment': json.dumps(analysis.get('next_experiment', {})),
                'risks': json.dumps(analysis.get('risks', [])),
                'expected_bitrate_mbps': Decimal(str(analysis.get('expected_bitrate_mbps', 0))),
                'confidence_score': Decimal(str(analysis.get('confidence_score', 0)))
            })
            
            logger.info(f"Reasoning logged: {reasoning_id}")
            
        except Exception as e:
            logger.error(f"Error logging reasoning: {e}")
    
    def generate_compression_code(self, analysis: Dict) -> Optional[Dict]:
        """
        Generate new compression algorithm code based on LLM analysis.
        
        Args:
            analysis: LLM analysis of past experiments
            
        Returns:
            Dict with generated code and metadata
        """
        if not self.client and not self.use_direct_api:
            logger.warning("LLM not available for code generation")
            return None
        
        try:
            # Build prompt with system context
            if self.system_prompt:
                # Use full system prompt
                prompt = f"""{self.system_prompt}

---

## CURRENT EXPERIMENT CONTEXT

Based on your analysis of past experiments, generate a NEW compression algorithm.

**Your Analysis:**
- Root Cause: {analysis.get('root_cause', '')}
- Hypothesis: {analysis.get('hypothesis', '')}
- Key Insights: {json.dumps(analysis.get('insights', []), indent=2)}

**Your Task:**
Generate the `compress_video_frame` function that implements your hypothesis.
Focus on continuous improvement - beat your previous iteration!

Generate ONLY the Python code (imports + function), no markdown explanations.
"""
            else:
                # Fallback to basic prompt if system prompt not loaded
                prompt = f"""Generate a Python video compression function based on this analysis.

ANALYSIS:
Root Cause: {analysis.get('root_cause', '')}
Hypothesis: {analysis.get('hypothesis', '')}
Insights: {json.dumps(analysis.get('insights', []))}

REQUIREMENTS:
1. Function signature: def compress_video_frame(frame: np.ndarray, frame_index: int, config: dict) -> bytes
2. Input: frame is numpy array (H, W, 3) uint8 RGB, frame_index is int, config has parameters
3. Output: compressed bytes for this frame
4. Use only: numpy, cv2, math, json, struct, base64, torch, torchvision
5. NO imports of: os, sys, subprocess, socket, requests, urllib
6. Focus on the hypothesis from the analysis
7. Be creative but practical - this will run on real video
8. Include comments explaining your approach

Generate ONLY the Python function code, no explanations outside the code.
Start with 'import' statements, then the function definition.
"""
            
            if self.client:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.8,
                    messages=[{"role": "user", "content": prompt}]
                )
                code = message.content[0].text
            else:
                code = self._call_claude_direct(prompt)
            
            if not code:
                return None
            
            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            logger.info("✅ Generated compression code")
            logger.info(f"Code length: {len(code)} characters")
            
            return {
                'code': code,
                'function_name': 'compress_video_frame',
                'generated_at': datetime.utcnow().isoformat(),
                'based_on_analysis': analysis.get('hypothesis', '')[:200]
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return None
    
    def plan_next_experiment(self) -> Dict:
        """
        Main method: Analyze past experiments and plan the next one.
        
        Returns:
            Dict with experiment plan and reasoning
        """
        logger.info("=" * 60)
        logger.info("LLM EXPERIMENT PLANNER - ANALYZING RESULTS")
        logger.info("=" * 60)
        
        # Fetch recent experiments
        experiments = self.analyze_recent_experiments(limit=5)
        
        if not experiments:
            logger.warning("No experiments found - using default plan")
            return {
                'status': 'no_data',
                'plan': 'run_baseline_experiment',
                'reasoning': 'No previous experiments to analyze'
            }
        
        # Get LLM analysis
        analysis = self.get_llm_analysis(experiments)
        
        if not analysis:
            logger.error("Failed to get analysis")
            return {
                'status': 'error',
                'plan': 'retry_previous',
                'reasoning': 'Analysis failed'
            }
        
        # Log reasoning
        latest_exp_id = experiments[0].get('experiment_id', 'unknown')
        self.log_reasoning(analysis, latest_exp_id)
        
        # Print analysis
        logger.info("\n" + "=" * 60)
        logger.info("ROOT CAUSE ANALYSIS")
        logger.info("=" * 60)
        logger.info(analysis.get('root_cause', 'N/A'))
        
        logger.info("\n" + "=" * 60)
        logger.info("KEY INSIGHTS")
        logger.info("=" * 60)
        for insight in analysis.get('insights', []):
            logger.info(f"• {insight}")
        
        logger.info("\n" + "=" * 60)
        logger.info("HYPOTHESIS")
        logger.info("=" * 60)
        logger.info(analysis.get('hypothesis', 'N/A'))
        
        logger.info("\n" + "=" * 60)
        logger.info("NEXT EXPERIMENT PLAN")
        logger.info("=" * 60)
        next_exp = analysis.get('next_experiment', {})
        logger.info(f"Approach: {next_exp.get('approach', 'N/A')}")
        logger.info("Changes:")
        for change in next_exp.get('changes', []):
            logger.info(f"  • {change}")
        
        logger.info("\n" + "=" * 60)
        logger.info("EXPECTED RESULTS")
        logger.info("=" * 60)
        logger.info(f"Target Bitrate: {analysis.get('expected_bitrate_mbps', 0):.2f} Mbps")
        logger.info(f"Confidence: {analysis.get('confidence_score', 0)*100:.1f}%")
        
        return {
            'status': 'success',
            'analysis': analysis,
            'latest_experiment_id': latest_exp_id,
            'timestamp': datetime.utcnow().isoformat()
        }


if __name__ == '__main__':
    """Test the LLM planner."""
    planner = LLMExperimentPlanner()
    result = planner.plan_next_experiment()
    
    print("\n" + "=" * 60)
    print("PLANNER RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))

