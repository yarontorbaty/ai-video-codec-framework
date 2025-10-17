#!/usr/bin/env python3
"""
Log Analyzer - Uses LLM to analyze experiment failure logs
"""

import json
import logging
import boto3
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class LogAnalyzer:
    """Analyzes experiment logs using LLM to extract failure reasons."""
    
    def __init__(self):
        """Initialize log analyzer with AWS clients."""
        self.secrets_client = boto3.client('secretsmanager', region_name='us-east-1')
        self.api_key = None
        
    def _get_api_key(self) -> Optional[str]:
        """Get Anthropic API key from Secrets Manager."""
        if self.api_key:
            return self.api_key
            
        try:
            response = self.secrets_client.get_secret_value(
                SecretId='ai-video-codec/anthropic-api-key'
            )
            secret_data = json.loads(response['SecretString'])
            self.api_key = secret_data.get('ANTHROPIC_API_KEY')
            return self.api_key
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None
    
    def analyze_failure(self, 
                       experiment_id: str,
                       experiment_type: str,
                       error_message: str,
                       logs: str,
                       code_snippet: Optional[str] = None) -> Dict:
        """
        Analyze experiment failure using LLM.
        
        Args:
            experiment_id: Experiment identifier
            experiment_type: Type of experiment (e.g., 'llm_generated_code_evolution')
            error_message: Error message or exception
            logs: Full logs from the experiment
            code_snippet: Optional code that failed (for LLM code failures)
            
        Returns:
            Dict with analysis results:
            {
                'failure_category': str,  # e.g., 'syntax_error', 'runtime_error', 'validation_error'
                'root_cause': str,        # Brief explanation
                'fix_suggestion': str,    # What to fix
                'severity': str,          # 'critical', 'high', 'medium', 'low'
            }
        """
        api_key = self._get_api_key()
        if not api_key:
            return self._fallback_analysis(error_message, logs)
        
        try:
            # Use requests library for direct API call
            import requests
            
            prompt = self._build_analysis_prompt(
                experiment_id, experiment_type, error_message, logs, code_snippet
            )
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json',
                },
                json={
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 1024,
                    'messages': [{
                        'role': 'user',
                        'content': prompt
                    }]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text']
                
                # Parse LLM response
                analysis = self._parse_llm_response(content)
                logger.info(f"✅ LLM analyzed failure for {experiment_id}")
                return analysis
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return self._fallback_analysis(error_message, logs)
                
        except Exception as e:
            logger.error(f"Failed to analyze logs with LLM: {e}")
            return self._fallback_analysis(error_message, logs)
    
    def _build_analysis_prompt(self, 
                               experiment_id: str,
                               experiment_type: str,
                               error_message: str,
                               logs: str,
                               code_snippet: Optional[str]) -> str:
        """Build prompt for LLM analysis."""
        
        # Truncate logs if too long (keep last 3000 chars for context)
        if len(logs) > 3000:
            logs = "...[truncated]...\n" + logs[-3000:]
        
        prompt = f"""Analyze this experiment failure and provide a concise diagnosis.

Experiment: {experiment_id}
Type: {experiment_type}
Error: {error_message}

Logs:
{logs}
"""
        
        if code_snippet:
            code_preview = code_snippet[:1000] if len(code_snippet) > 1000 else code_snippet
            prompt += f"""
Code that failed:
```python
{code_preview}
```
"""
        
        prompt += """
Provide analysis in this EXACT JSON format:
{
  "failure_category": "<one of: syntax_error, import_error, runtime_error, validation_error, logic_error, timeout, resource_error>",
  "root_cause": "<brief 1-2 sentence explanation>",
  "fix_suggestion": "<specific actionable fix>",
  "severity": "<one of: critical, high, medium, low>"
}

Only respond with the JSON, no other text."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM's JSON response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            analysis = json.loads(response)
            
            # Validate required fields
            required = ['failure_category', 'root_cause', 'fix_suggestion', 'severity']
            if all(field in analysis for field in required):
                return analysis
            else:
                logger.warning("LLM response missing required fields")
                return self._fallback_analysis("", "")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response[:200]}")
            return self._fallback_analysis("", "")
    
    def _fallback_analysis(self, error_message: str, logs: str) -> Dict:
        """Provide basic analysis without LLM."""
        
        error_lower = error_message.lower()
        logs_lower = logs.lower()
        
        # Pattern matching for common errors
        if 'forbidden import' in error_lower or 'forbidden attribute' in error_lower:
            return {
                'failure_category': 'validation_error',
                'root_cause': 'Code validation failed due to security restrictions (forbidden imports or attributes)',
                'fix_suggestion': 'Use only allowed libraries: numpy, cv2, torch, torchvision, json, struct, base64',
                'severity': 'high'
            }
        elif 'syntaxerror' in error_lower or 'indentationerror' in error_lower:
            return {
                'failure_category': 'syntax_error',
                'root_cause': 'Python syntax error in generated code',
                'fix_suggestion': 'Fix syntax errors: check indentation, brackets, and Python syntax',
                'severity': 'high'
            }
        elif 'importerror' in error_lower or 'modulenotfounderror' in error_lower:
            return {
                'failure_category': 'import_error',
                'root_cause': 'Failed to import required module',
                'fix_suggestion': 'Check that module is installed and allowed in sandbox',
                'severity': 'high'
            }
        elif 'timeout' in error_lower or 'exceeded' in logs_lower:
            return {
                'failure_category': 'timeout',
                'root_cause': 'Execution exceeded time limit',
                'fix_suggestion': 'Optimize algorithm or reduce computational complexity',
                'severity': 'medium'
            }
        elif 'memoryerror' in error_lower or 'out of memory' in logs_lower:
            return {
                'failure_category': 'resource_error',
                'root_cause': 'Insufficient memory for operation',
                'fix_suggestion': 'Reduce memory usage or use more efficient data structures',
                'severity': 'high'
            }
        elif 'not better than' in logs_lower or 'not adopted' in logs_lower:
            return {
                'failure_category': 'logic_error',
                'root_cause': 'Code executed but performance did not meet adoption criteria',
                'fix_suggestion': 'Improve compression algorithm to achieve better bitrate/quality trade-off',
                'severity': 'low'
            }
        else:
            return {
                'failure_category': 'runtime_error',
                'root_cause': f'Execution failed: {error_message[:100]}',
                'fix_suggestion': 'Review logs and fix runtime errors',
                'severity': 'medium'
            }


def test_analyzer():
    """Test log analyzer."""
    analyzer = LogAnalyzer()
    
    # Test 1: Validation error
    result = analyzer.analyze_failure(
        experiment_id='test_1',
        experiment_type='llm_generated_code',
        error_message='Forbidden import from: torchvision',
        logs='Code validation failed\nForbidden import from: torchvision; Forbidden attribute: eval',
    )
    print("Test 1 (validation):", json.dumps(result, indent=2))
    
    # Test 2: Syntax error
    result = analyzer.analyze_failure(
        experiment_id='test_2',
        experiment_type='llm_generated_code',
        error_message='SyntaxError: invalid syntax',
        logs='  File "codec.py", line 42\n    if x = 5:\n       ^\nSyntaxError: invalid syntax',
    )
    print("Test 2 (syntax):", json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_analyzer()

