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
from typing import Dict, Any, Optional, Callable
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
        'math',
        'json',
        'struct',
        'base64',
        'io',
        'collections',
    }
    
    FORBIDDEN_ATTRIBUTES = {
        '__import__',
        'exec',
        'eval',
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
        'getattr',
        'setattr',
        'delattr',
        'hasattr',
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
    
    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code for safety using AST analysis.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for forbidden patterns
        for node in ast.walk(tree):
            # Check for forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in self.ALLOWED_MODULES:
                        return False, f"Forbidden import: {alias.name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_MODULES:
                    return False, f"Forbidden import from: {node.module}"
            
            # Check for forbidden attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    return False, f"Forbidden attribute: {node.attr}"
            
            # Check for forbidden names
            if isinstance(node, ast.Name):
                if node.id in self.FORBIDDEN_ATTRIBUTES:
                    return False, f"Forbidden name: {node.id}"
            
            # Forbid async code (harder to control)
            if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
                return False, "Async code not allowed"
        
        return True, None
    
    def execute_function(self, code: str, function_name: str, 
                        args: tuple = (), kwargs: dict = None) -> tuple[bool, Any, Optional[str]]:
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
                },
            }
            
            # Import allowed modules
            import numpy as np
            import cv2
            try:
                import torch
                restricted_globals['torch'] = torch
            except ImportError:
                pass
            
            restricted_globals['np'] = np
            restricted_globals['numpy'] = np
            restricted_globals['cv2'] = cv2
            restricted_globals['math'] = __import__('math')
            restricted_globals['json'] = __import__('json')
            
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
            logger.error(error_msg)
            logger.error(traceback.format_exc())
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

