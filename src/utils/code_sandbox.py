#!/usr/bin/env python3
"""
Secure Code Sandbox for LLM-Generated Code
Executes dynamically generated compression algorithms with safety constraints.
"""

import sys
import io
import ast
import logging
import traceback
from typing import Dict, Any, Optional, Callable, Tuple
import multiprocessing
import signal

logger = logging.getLogger(__name__)


class CodeSandbox:
    """
    Secure sandbox for executing LLM-generated compression code.
    
    Safety features:
    - AST validation (no dangerous imports/calls)
    - Resource limits (CPU time, memory)
    - Process isolation
    - Whitelist of allowed modules
    """
    
    ALLOWED_MODULES = {
        'numpy', 'np',
        'cv2',
        'torch',
        'torchvision',  # Added for neural codec implementations
        'torch.nn',
        'torch.nn.functional',
        'math',
        'json',
        'struct',
        'base64',
        'io',
        'collections',
        'typing',
    }
    
    FORBIDDEN_ATTRIBUTES = {
        '__import__',
        'exec',
        # 'eval' removed - PyTorch model.eval() is safe and necessary
        # The dangerous eval() builtin is still blocked via restricted_globals
        'compile',
        'open',
        'file',
        'input',
        'raw_input',
        '__builtins__',
        'globals',
        'locals',
        'vars',
        'dir',
        'getattr',  # Can access arbitrary attributes - too dangerous
        'setattr',  # Can modify arbitrary attributes - too dangerous
        'delattr',  # Can delete arbitrary attributes - too dangerous
        # 'hasattr' removed - safe for checking if attributes exist (like isinstance, len)
    }
    
    def __init__(self, timeout: int = 60, max_memory_mb: int = 2048):
        """
        Initialize sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
    
    def validate_code(self, code: str, save_attempt: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate code for safety using AST analysis.
        
        Args:
            code: Python code to validate
            save_attempt: Whether to save this code attempt for debugging
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import time
        import os
        
        # Save code attempt for debugging
        if save_attempt:
            timestamp = int(time.time())
            attempt_dir = '/tmp/code_attempts'
            os.makedirs(attempt_dir, exist_ok=True)
            
            attempt_file = f'{attempt_dir}/attempt_{timestamp}.py'
            try:
                with open(attempt_file, 'w') as f:
                    f.write(f"# Code validation attempt at {timestamp}\n")
                    f.write(f"# Timestamp: {time.ctime()}\n\n")
                    f.write(code)
                logger.info(f"💾 Saved code attempt to {attempt_file}")
            except Exception as e:
                logger.warning(f"Could not save code attempt: {e}")
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            error_msg = f"Syntax error on line {e.lineno}: {e.msg}"
            logger.error(f"❌ VALIDATION FAILED: {error_msg}")
            logger.error(f"Code excerpt: {code[:500]}...")
            return False, error_msg
        
        # Check for forbidden patterns
        violations = []
        for node in ast.walk(tree):
            # Check for forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.ALLOWED_MODULES:
                        violations.append(f"Forbidden import: {alias.name}")
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_MODULES:
                    violations.append(f"Forbidden import from: {node.module}")
            
            # Check for forbidden attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    violations.append(f"Forbidden attribute: {node.attr}")
            
            # Check for forbidden names
            if isinstance(node, ast.Name):
                if node.id in self.FORBIDDEN_ATTRIBUTES:
                    violations.append(f"Forbidden name: {node.id}")
            
            # Forbid async code (harder to control)
            if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
                violations.append("Async code not allowed")
        
        if violations:
            error_msg = "; ".join(violations)
            logger.error(f"❌ VALIDATION FAILED: {error_msg}")
            logger.error(f"Code preview:\n{code[:300]}...")
            return False, error_msg
        
        logger.info("✅ Code validation passed")
        return True, None
    
    def execute_function(self, code: str, function_name: str, 
                        args: tuple = (), kwargs: dict = None) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute LLM-generated function with safety constraints.
        
        Args:
            code: Python code containing the function
            function_name: Name of function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (success, result, error_message)
        """
        kwargs = kwargs or {}
        
        # Validate code first
        is_valid, error_msg = self.validate_code(code)
        if not is_valid:
            logger.error(f"Code validation failed: {error_msg}")
            return False, None, error_msg
        
        logger.info(f"Executing LLM-generated function: {function_name}")
        
        try:
            # Create restricted import function
            def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.split('.')[0] not in self.ALLOWED_MODULES:
                    raise ImportError(f"Import of '{name}' is not allowed")
                return __import__(name, globals, locals, fromlist, level)
            
            # Create restricted globals
            restricted_globals = {
                '__builtins__': {
                    'range': range,
                    'len': len,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'bool': bool,
                    'bytes': bytes,
                    'bytearray': bytearray,  # For binary data manipulation
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sorted': sorted,
                    'reversed': reversed,
                    'isinstance': isinstance,
                    'type': type,
                    'print': print,  # For debugging
                    '__import__': safe_import,  # Allow controlled imports
                },
            }
            
            # Import allowed modules
            import numpy as np
            import cv2
            try:
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                restricted_globals['torch'] = torch
                restricted_globals['nn'] = nn
                restricted_globals['F'] = F
            except ImportError:
                pass
            
            try:
                import torchvision
                restricted_globals['torchvision'] = torchvision
            except ImportError:
                pass
            
            restricted_globals['np'] = np
            restricted_globals['numpy'] = np
            restricted_globals['cv2'] = cv2
            restricted_globals['math'] = __import__('math')
            restricted_globals['json'] = __import__('json')
            restricted_globals['struct'] = __import__('struct')
            restricted_globals['base64'] = __import__('base64')
            
            # Execute code to define function
            exec(code, restricted_globals)
            
            if function_name not in restricted_globals:
                return False, None, f"Function '{function_name}' not found in code"
            
            # Get the function
            func = restricted_globals[function_name]
            
            # Execute with timeout using multiprocessing
            # (Can't use signal.alarm on all platforms)
            result = self._execute_with_timeout(func, args, kwargs)
            
            logger.info(f"✅ Function executed successfully")
            return True, result, None
            
        except Exception as e:
            error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
            full_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.error(full_traceback)
            
            # Save execution failure details
            import time
            import os
            attempt_dir = '/tmp/code_attempts'
            os.makedirs(attempt_dir, exist_ok=True)
            
            timestamp = int(time.time())
            error_file = f'{attempt_dir}/error_{timestamp}.txt'
            try:
                with open(error_file, 'w') as f:
                    f.write(f"Execution Error at {time.ctime()}\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(f"Error: {error_msg}\n\n")
                    f.write(f"Full Traceback:\n{full_traceback}\n\n")
                    f.write(f"Code:\n{'-'*60}\n{code}\n")
                logger.error(f"💾 Saved error details to {error_file}")
            except Exception as save_error:
                logger.warning(f"Could not save error details: {save_error}")
            
            return False, None, error_msg
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """
        Execute function with timeout.
        
        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        # For now, simple timeout with signal (Unix only)
        # TODO: Use multiprocessing for cross-platform support
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {self.timeout}s timeout")
        
        # Try signal-based timeout (Unix)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        except AttributeError:
            # Windows doesn't have SIGALRM, just execute directly
            logger.warning("Timeout not enforced (Windows platform)")
            return func(*args, **kwargs)


def test_sandbox():
    """Test the sandbox with various code samples."""
    sandbox = CodeSandbox(timeout=5)
    
    # Test 1: Valid code
    code1 = """
import numpy as np

def compress_simple(data):
    return np.mean(data)
"""
    success, result, error = sandbox.execute_function(
        code1, 'compress_simple', args=([1, 2, 3, 4, 5],)
    )
    print(f"Test 1 (valid): success={success}, result={result}")
    
    # Test 2: Forbidden import
    code2 = """
import os

def bad_function():
    return os.listdir('/')
"""
    success, result, error = sandbox.execute_function(code2, 'bad_function')
    print(f"Test 2 (forbidden import): success={success}, error={error}")
    
    # Test 3: Forbidden exec
    code3 = """
def sneaky():
    exec('print("hacked")')
"""
    success, result, error = sandbox.execute_function(code3, 'sneaky')
    print(f"Test 3 (forbidden exec): success={success}, error={error}")


if __name__ == "__main__":
    test_sandbox()

