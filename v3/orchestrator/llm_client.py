"""
V3.0 LLM Client - Claude API Wrapper

Generates video compression code using Anthropic's Claude API
"""

import anthropic
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Interface to Claude API for code generation"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def generate_compression_code(
        self,
        iteration: int,
        previous_results: List[Dict]
    ) -> Optional[Dict[str, str]]:
        """
        Generate encoding and decoding code
        
        Returns:
            {
                'encoding': str,  # Python code
                'decoding': str,  # Python code
                'reasoning': str  # LLM's explanation
            }
        """
        try:
            prompt = self._build_prompt(iteration, previous_results)
            
            logger.info(f"🤖 Calling Claude API (iteration {iteration})...")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            content = response.content[0].text
            
            # Parse response
            code = self._parse_response(content)
            
            if code:
                logger.info(f"✅ Code generation successful")
                return code
            else:
                logger.error(f"❌ Failed to parse LLM response")
                return None
                
        except Exception as e:
            logger.error(f"❌ LLM API error: {e}", exc_info=True)
            return None
    
    def _build_prompt(self, iteration: int, previous_results: List[Dict]) -> str:
        """Build prompt for code generation"""
        
        prompt = f"""You are an expert video compression engineer. Generate Python code for a video compression algorithm.

**Iteration {iteration}**

**Task:**
Create two Python functions:
1. `run_encoding_agent(frames, output_path)` - Compress a list of video frames
2. `run_decoding_agent(input_path, output_path, frame_count)` - Decompress back to video

**Requirements:**
- The encoder must create a compressed file at `output_path`
- The decoder must create a video file at `output_path`
- Use only: cv2, numpy (no torch, no tensorflow)
- Focus on REAL compression (not procedural generation)
- Target: PSNR > 30dB, SSIM > 0.85, compression ratio > 10x

**Input:**
- `frames`: List of numpy arrays (BGR images, 640x480)
- `output_path`: Where to save compressed data
- `frame_count`: Number of frames to decode

"""
        
        # Add previous results for evolution
        if previous_results:
            prompt += "\n**Previous Results:**\n"
            for i, result in enumerate(previous_results[-3:], 1):  # Last 3
                metrics = result.get('metrics', {})
                prompt += f"\nExperiment {i}:\n"
                prompt += f"- PSNR: {metrics.get('psnr_db', 0):.2f} dB\n"
                prompt += f"- SSIM: {metrics.get('ssim', 0):.3f}\n"
                prompt += f"- Compression: {metrics.get('compression_ratio', 0):.1f}x\n"
                if result.get('status') == 'failed':
                    prompt += f"- Error: {result.get('error', 'Unknown')}\n"
            
            prompt += "\n**Your Goal:** Improve upon the previous results.\n"
        else:
            prompt += "\n**Your Goal:** Create a working baseline compression algorithm.\n"
        
        prompt += """
**Response Format:**
Provide your code in this exact format:

```python
# ENCODING
def run_encoding_agent(frames, output_path):
    # Your encoding code here
    pass
```

```python
# DECODING  
def run_decoding_agent(input_path, output_path, frame_count):
    # Your decoding code here
    pass
```

**REASONING:** (Explain your approach in 2-3 sentences)
"""
        
        return prompt
    
    def _parse_response(self, content: str) -> Optional[Dict[str, str]]:
        """Parse LLM response to extract code"""
        try:
            encoding = None
            decoding = None
            reasoning = ""
            
            # Extract encoding code
            if "# ENCODING" in content:
                start = content.find("# ENCODING")
                end = content.find("```", start + 100)  # Find next ```
                if end > start:
                    encoding = content[start:end].strip()
            
            # Extract decoding code
            if "# DECODING" in content:
                start = content.find("# DECODING")
                end = content.find("```", start + 100)
                if end > start:
                    decoding = content[start:end].strip()
            
            # Extract reasoning
            if "REASONING:" in content:
                start = content.find("REASONING:") + len("REASONING:")
                reasoning = content[start:].strip()
            
            if encoding and decoding:
                return {
                    'encoding': encoding,
                    'decoding': decoding,
                    'reasoning': reasoning[:500]  # Limit length
                }
            
            # Fallback: try to extract any Python code blocks
            import re
            code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
            if len(code_blocks) >= 2:
                return {
                    'encoding': code_blocks[0],
                    'decoding': code_blocks[1],
                    'reasoning': 'Extracted from code blocks'
                }
            
            logger.error(f"Could not parse response. Content:\n{content[:500]}...")
            return None
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

