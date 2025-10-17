#!/usr/bin/env python3
"""
Framework Modifier - Allows LLM to safely modify its own code
Provides tools for the LLM to fix bugs, improve code, and self-heal
"""

import os
import json
import shutil
import logging
import subprocess
import shlex
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class FrameworkModifier:
    """
    Safely applies LLM-requested modifications to the framework itself.
    Includes backup, rollback, and validation mechanisms.
    """
    
    def __init__(self, base_path: str = "/home/ec2-user/ai-video-codec"):
        """
        Initialize framework modifier.
        
        Args:
            base_path: Root directory of the framework
        """
        self.base_path = base_path
        self.backup_dir = os.path.join(base_path, '.framework_backups')
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Track modifications for rollback
        self.modification_history = []
        
        # Git configuration for autonomous commits
        self.git_branch = "self-improved-framework"
        self.git_enabled = True
        self.git_user_name = "AI Video Codec Agent"
        self.git_user_email = "ai-agent@ai-video-codec.autonomo.us"
        
        # Allowed file patterns for modification
        self.allowed_patterns = [
            'src/**/*.py',
            'scripts/**/*.py',
            'scripts/**/*.sh',
            'requirements.txt',
            'LLM_SYSTEM_PROMPT.md',
        ]
        
        # Protected files that need extra validation
        self.protected_files = [
            'scripts/autonomous_orchestrator_llm.sh',
            'src/agents/llm_experiment_planner.py',
        ]
        
        # Initialize git config
        self._init_git_config()
    
    def _init_git_config(self):
        """Initialize git configuration for autonomous commits."""
        if not self.git_enabled:
            return
        
        try:
            # Configure git user for commits
            subprocess.run(
                f"cd {self.base_path} && git config user.name '{self.git_user_name}'",
                shell=True,
                capture_output=True,
                check=False
            )
            subprocess.run(
                f"cd {self.base_path} && git config user.email '{self.git_user_email}'",
                shell=True,
                capture_output=True,
                check=False
            )
            
            # Ensure we're on the self-improvement branch
            result = subprocess.run(
                f"cd {self.base_path} && git rev-parse --abbrev-ref HEAD",
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            current_branch = result.stdout.strip()
            if current_branch != self.git_branch:
                logger.info(f"Switching to {self.git_branch} branch for autonomous modifications...")
                # Fetch latest
                subprocess.run(
                    f"cd {self.base_path} && git fetch origin",
                    shell=True,
                    capture_output=True,
                    check=False
                )
                # Checkout or create branch
                subprocess.run(
                    f"cd {self.base_path} && git checkout {self.git_branch} || git checkout -b {self.git_branch}",
                    shell=True,
                    capture_output=True,
                    check=False
                )
                # Pull latest if branch exists on remote
                subprocess.run(
                    f"cd {self.base_path} && git pull origin {self.git_branch}",
                    shell=True,
                    capture_output=True,
                    check=False
                )
            
            logger.info(f"✅ Git configured for autonomous commits to {self.git_branch}")
            
        except Exception as e:
            logger.warning(f"⚠️  Git configuration failed: {e}")
            self.git_enabled = False
    
    def _git_commit_and_push(self, file_path: str, reason: str) -> bool:
        """
        Commit and push a file modification to the self-improvement branch.
        
        Args:
            file_path: Path to the modified file
            reason: Reason for the modification
            
        Returns:
            True if successful, False otherwise
        """
        if not self.git_enabled:
            logger.warning("Git commits disabled")
            return False
        
        try:
            # Stage the file
            result = subprocess.run(
                f"cd {self.base_path} && git add {file_path}",
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to stage {file_path}: {result.stderr}")
                return False
            
            # Commit with detailed message
            commit_msg = f"🤖 LLM: {reason}\n\nFile: {file_path}\nTimestamp: {datetime.utcnow().isoformat()}\nAgent: AI Video Codec Autonomous Framework Modifier"
            
            result = subprocess.run(
                f"cd {self.base_path} && git commit -m {shlex.quote(commit_msg)}",
                shell=True,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                # Check if it's just "nothing to commit"
                if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                    logger.info("No changes to commit")
                    return True
                logger.error(f"Failed to commit: {result.stderr}")
                return False
            
            logger.info(f"✅ Committed: {file_path}")
            
            # Push to remote
            result = subprocess.run(
                f"cd {self.base_path} && git push origin {self.git_branch}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to push: {result.stderr}")
                return False
            
            logger.info(f"✅ Pushed to origin/{self.git_branch}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Git push timed out")
            return False
        except Exception as e:
            logger.error(f"Git commit/push error: {e}")
            return False
    
    def is_file_modifiable(self, file_path: str) -> Tuple[bool, str]:
        """
        Check if a file can be safely modified.
        
        Returns:
            (allowed, reason)
        """
        abs_path = os.path.join(self.base_path, file_path)
        
        # Check if file exists
        if not os.path.exists(abs_path):
            return False, f"File does not exist: {file_path}"
        
        # Check if in allowed patterns
        import fnmatch
        allowed = False
        for pattern in self.allowed_patterns:
            if fnmatch.fnmatch(file_path, pattern):
                allowed = True
                break
        
        if not allowed:
            return False, f"File not in allowed patterns: {file_path}"
        
        # Check if file is too large (safety limit: 100KB)
        file_size = os.path.getsize(abs_path)
        if file_size > 100 * 1024:
            return False, f"File too large ({file_size} bytes)"
        
        return True, "OK"
    
    def backup_file(self, file_path: str) -> str:
        """
        Create a backup of a file before modification.
        
        Returns:
            Backup file path
        """
        abs_path = os.path.join(self.base_path, file_path)
        timestamp = int(time.time())
        
        # Create backup with timestamp
        backup_name = f"{file_path.replace('/', '_')}_{timestamp}.backup"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        shutil.copy2(abs_path, backup_path)
        logger.info(f"✅ Backed up {file_path} to {backup_path}")
        
        return backup_path
    
    def modify_file(self, 
                   file_path: str, 
                   modification_type: str,
                   content: str,
                   reason: str) -> Dict:
        """
        Modify a framework file.
        
        Args:
            file_path: Relative path from base_path
            modification_type: 'replace_content', 'append', 'search_replace'
            content: New content or search/replace data
            reason: Why this modification is needed
            
        Returns:
            Result dict with success status
        """
        try:
            # Validate file can be modified
            allowed, msg = self.is_file_modifiable(file_path)
            if not allowed:
                return {
                    'success': False,
                    'error': msg,
                    'file': file_path
                }
            
            abs_path = os.path.join(self.base_path, file_path)
            
            # Create backup
            backup_path = self.backup_file(file_path)
            
            # Read current content
            with open(abs_path, 'r') as f:
                original_content = f.read()
            
            # Apply modification based on type
            if modification_type == 'replace_content':
                new_content = content
                
            elif modification_type == 'append':
                new_content = original_content + '\n' + content
                
            elif modification_type == 'search_replace':
                # Expect content to be JSON: {"search": "...", "replace": "..."}
                data = json.loads(content) if isinstance(content, str) else content
                search = data.get('search', '')
                replace = data.get('replace', '')
                
                if search not in original_content:
                    return {
                        'success': False,
                        'error': f"Search string not found in {file_path}",
                        'file': file_path
                    }
                
                new_content = original_content.replace(search, replace, 1)
            
            else:
                return {
                    'success': False,
                    'error': f"Unknown modification type: {modification_type}",
                    'file': file_path
                }
            
            # Write new content
            with open(abs_path, 'w') as f:
                f.write(new_content)
            
            # Record modification
            mod_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'file': file_path,
                'type': modification_type,
                'reason': reason,
                'backup': backup_path,
                'success': True
            }
            self.modification_history.append(mod_record)
            
            logger.info(f"✅ Modified {file_path}: {reason}")
            
            # Automatically commit and push to self-improved-framework branch
            git_success = self._git_commit_and_push(file_path, reason)
            if git_success:
                logger.info(f"✅ Changes committed to {self.git_branch}")
                mod_record['git_committed'] = True
                mod_record['git_branch'] = self.git_branch
            else:
                logger.warning(f"⚠️  Git commit failed - changes are local only")
                mod_record['git_committed'] = False
            
            return {
                'success': True,
                'file': file_path,
                'backup': backup_path,
                'modification': mod_record,
                'git_committed': git_success
            }
            
        except Exception as e:
            logger.error(f"❌ Error modifying {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': file_path
            }
    
    def rollback_file(self, file_path: str) -> Dict:
        """
        Rollback a file to its last backup.
        
        Returns:
            Result dict
        """
        try:
            # Find most recent backup
            backups = [m for m in self.modification_history if m['file'] == file_path]
            if not backups:
                return {
                    'success': False,
                    'error': f"No backup found for {file_path}"
                }
            
            last_backup = backups[-1]['backup']
            abs_path = os.path.join(self.base_path, file_path)
            
            # Restore from backup
            shutil.copy2(last_backup, abs_path)
            
            logger.info(f"✅ Rolled back {file_path} from {last_backup}")
            
            return {
                'success': True,
                'file': file_path,
                'restored_from': last_backup
            }
            
        except Exception as e:
            logger.error(f"❌ Error rolling back {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': file_path
            }
    
    def run_command(self, 
                   command: str,
                   reason: str,
                   timeout: int = 60) -> Dict:
        """
        Execute a shell command.
        
        Args:
            command: Shell command to run
            reason: Why this command is needed
            timeout: Max execution time in seconds
            
        Returns:
            Result dict
        """
        try:
            # Security checks
            dangerous_patterns = ['rm -rf', 'dd if=', '> /dev/', 'format', 'mkfs']
            for pattern in dangerous_patterns:
                if pattern in command.lower():
                    return {
                        'success': False,
                        'error': f"Dangerous command pattern detected: {pattern}",
                        'command': command
                    }
            
            logger.info(f"🔧 Running command: {command}")
            logger.info(f"   Reason: {reason}")
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = (result.returncode == 0)
            
            if success:
                logger.info(f"✅ Command succeeded")
            else:
                logger.error(f"❌ Command failed with code {result.returncode}")
            
            return {
                'success': success,
                'command': command,
                'reason': reason,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timed out after {timeout}s",
                'command': command
            }
        except Exception as e:
            logger.error(f"❌ Error running command: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command
            }
    
    def install_package(self, package: str, reason: str) -> Dict:
        """
        Install a Python package.
        
        Args:
            package: Package name (e.g., 'numpy==1.21.0')
            reason: Why this package is needed
            
        Returns:
            Result dict
        """
        command = f"pip3 install {package}"
        return self.run_command(command, f"Install {package}: {reason}")
    
    def restart_orchestrator(self) -> Dict:
        """
        Restart the orchestrator service to apply changes.
        
        Returns:
            Result dict
        """
        try:
            logger.info("🔄 Restarting orchestrator...")
            
            # Kill current orchestrator process
            subprocess.run("pkill -f autonomous_orchestrator_llm.sh", shell=True)
            time.sleep(2)
            
            # Start new orchestrator
            cmd = "cd /home/ec2-user/ai-video-codec && nohup bash scripts/autonomous_orchestrator_llm.sh > /tmp/orch.log 2>&1 &"
            subprocess.run(cmd, shell=True)
            time.sleep(3)
            
            # Verify it started
            check = subprocess.run(
                "pgrep -f autonomous_orchestrator_llm.sh",
                shell=True,
                capture_output=True
            )
            
            if check.returncode == 0:
                logger.info("✅ Orchestrator restarted successfully")
                return {
                    'success': True,
                    'message': 'Orchestrator restarted'
                }
            else:
                logger.error("❌ Orchestrator failed to restart")
                return {
                    'success': False,
                    'error': 'Failed to restart orchestrator'
                }
                
        except Exception as e:
            logger.error(f"❌ Error restarting orchestrator: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_modification_history(self) -> List[Dict]:
        """Get history of all modifications."""
        return self.modification_history


# Define tools for LLM function calling
FRAMEWORK_TOOLS = [
    {
        "name": "modify_framework_file",
        "description": "Modify a source file in the framework (Python, shell scripts, config files). Use this to fix bugs, improve code, or add features. Files are automatically backed up before modification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path to file (e.g., 'src/utils/code_sandbox.py')"
                },
                "modification_type": {
                    "type": "string",
                    "enum": ["replace_content", "append", "search_replace"],
                    "description": "Type of modification: replace_content (full file), append (add to end), search_replace (find and replace)"
                },
                "content": {
                    "type": "string",
                    "description": "New content, text to append, or JSON with {search: '', replace: ''}"
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation of why this change is needed"
                }
            },
            "required": ["file_path", "modification_type", "content", "reason"]
        }
    },
    {
        "name": "run_shell_command",
        "description": "Execute a shell command on the orchestrator. Use for system operations like installing packages, checking status, or running scripts. Dangerous commands are blocked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "reason": {
                    "type": "string",
                    "description": "Why this command needs to run"
                }
            },
            "required": ["command", "reason"]
        }
    },
    {
        "name": "install_python_package",
        "description": "Install a Python package using pip3. Use when you need a library that's not currently available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package name, optionally with version (e.g., 'requests==2.28.0')"
                },
                "reason": {
                    "type": "string",
                    "description": "Why this package is needed"
                }
            },
            "required": ["package", "reason"]
        }
    },
    {
        "name": "restart_orchestrator",
        "description": "Restart the orchestrator service to apply framework changes. Use after modifying code that affects the orchestrator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why restart is needed"
                }
            },
            "required": ["reason"]
        }
    },
    {
        "name": "rollback_file",
        "description": "Rollback a file to its last backup. Use if a modification caused issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to file to rollback"
                },
                "reason": {
                    "type": "string",
                    "description": "Why rollback is needed"
                }
            },
            "required": ["file_path", "reason"]
        }
    }
]

