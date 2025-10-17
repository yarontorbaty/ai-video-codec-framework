#!/usr/bin/env python3
"""
GitHub Integration for LLM Autonomous Commits
Enables the LLM to commit and push code improvements to GitHub.
"""

import os
import json
import logging
import subprocess
import boto3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GitHubIntegration:
    """
    Manages GitHub integration for autonomous LLM commits.
    """
    
    def __init__(self):
        self.secretsmanager = boto3.client('secretsmanager', region_name='us-east-1')
        self.credentials = None
        self.repo_path = '/home/ec2-user/ai-video-codec'
        
    def _get_credentials(self) -> Dict:
        """Fetch GitHub credentials from AWS Secrets Manager."""
        if self.credentials:
            return self.credentials
        
        try:
            response = self.secretsmanager.get_secret_value(
                SecretId='ai-video-codec/github-credentials'
            )
            self.credentials = json.loads(response['SecretString'])
            logger.info("✅ Retrieved GitHub credentials from Secrets Manager")
            return self.credentials
        except Exception as e:
            logger.error(f"Failed to retrieve GitHub credentials: {e}")
            raise
    
    def _setup_git_config(self):
        """Configure git with user details."""
        creds = self._get_credentials()
        
        try:
            subprocess.run(
                ['git', 'config', 'user.name', creds['username']],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ['git', 'config', 'user.email', creds['email']],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            logger.info("✅ Git config set up")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup git config: {e}")
            raise
    
    def _setup_remote(self):
        """Setup remote URL with authentication."""
        creds = self._get_credentials()
        
        # Create authenticated URL - use token only (PAT authentication)
        # Format: https://TOKEN@github.com/user/repo.git
        base_repo = creds['repo']
        
        # Remove any existing auth from URL
        if '@' in base_repo:
            # Extract just the github.com/user/repo part
            base_repo = 'https://' + base_repo.split('@', 1)[1]
        
        # Remove https:// prefix if present
        if base_repo.startswith('https://'):
            base_repo = base_repo[8:]
        elif base_repo.startswith('http://'):
            base_repo = base_repo[7:]
        
        # Construct clean authenticated URL
        repo_url = f"https://{creds['token']}@{base_repo}"
        
        try:
            # Check if remote exists
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Update existing remote
                subprocess.run(
                    ['git', 'remote', 'set-url', 'origin', repo_url],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
            else:
                # Add new remote
                subprocess.run(
                    ['git', 'remote', 'add', 'origin', repo_url],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
            
            logger.info("✅ Git remote configured")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup git remote: {e}")
            raise
    
    def initialize_repo(self) -> bool:
        """
        Initialize git repository if needed.
        Returns True if successful.
        """
        try:
            # Check if .git exists
            if not os.path.exists(os.path.join(self.repo_path, '.git')):
                logger.info("Initializing new git repository...")
                subprocess.run(
                    ['git', 'init'],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
            
            self._setup_git_config()
            self._setup_remote()
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize repo: {e}")
            return False
    
    def commit_code_evolution(self, 
                             version: int,
                             files_changed: List[str],
                             metrics: Dict,
                             description: str) -> Tuple[bool, Optional[str]]:
        """
        Commit code evolution changes to git.
        
        Args:
            version: Codec version number
            files_changed: List of file paths that changed
            metrics: Performance metrics for this version
            description: Human-readable description of changes
            
        Returns:
            Tuple of (success, commit_hash)
        """
        try:
            self.initialize_repo()
            
            # Stage files
            for file_path in files_changed:
                full_path = os.path.join(self.repo_path, file_path)
                if os.path.exists(full_path):
                    subprocess.run(
                        ['git', 'add', file_path],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True
                    )
                    logger.info(f"Staged: {file_path}")
            
            # Create commit message
            commit_message = self._generate_commit_message(version, metrics, description)
            
            # Commit
            result = subprocess.run(
                ['git', 'commit', '-m', commit_message],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Get commit hash
            hash_result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            commit_hash = hash_result.stdout.strip()
            
            logger.info(f"✅ Created commit: {commit_hash[:8]}")
            logger.info(f"📝 Message: {commit_message.split(chr(10))[0]}")
            
            return True, commit_hash
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Failed to commit: {error_msg}")
            return False, None
    
    def _generate_commit_message(self, version: int, metrics: Dict, description: str) -> str:
        """Generate detailed commit message for LLM evolution."""
        timestamp = datetime.utcnow().isoformat()
        
        message = f"🤖 Autonomous Code Evolution - v{version}\n\n"
        message += f"{description}\n\n"
        message += f"Performance Metrics:\n"
        
        if 'bitrate_mbps' in metrics:
            message += f"  • Bitrate: {metrics['bitrate_mbps']:.2f} Mbps\n"
        if 'compression_ratio' in metrics:
            message += f"  • Compression: {metrics['compression_ratio']:.2f}x\n"
        if 'improvement' in metrics:
            message += f"  • Improvement: {metrics['improvement']}\n"
        
        message += f"\nTimestamp: {timestamp}\n"
        message += f"Evolved by: LLM Autonomous System\n"
        
        return message
    
    def push_changes(self, branch: str = 'main') -> Tuple[bool, Optional[str]]:
        """
        Push committed changes to remote repository.
        
        Args:
            branch: Branch name to push to
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self.initialize_repo()
            
            # Check current branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            
            # If we're not on the target branch, try to checkout or create it
            if current_branch != branch:
                logger.info(f"Current branch: {current_branch}, switching to {branch}...")
                try:
                    # Try to checkout existing branch
                    subprocess.run(
                        ['git', 'checkout', branch],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError:
                    # Branch doesn't exist, rename current branch
                    subprocess.run(
                        ['git', 'branch', '-M', branch],
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True
                    )
                    logger.info(f"Renamed branch to {branch}")
            
            # Pull latest changes first (rebase to avoid merge commits)
            logger.info("Pulling latest changes...")
            try:
                subprocess.run(
                    ['git', 'pull', '--rebase', 'origin', branch],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True,
                    timeout=30
                )
            except subprocess.CalledProcessError:
                # If pull fails, it might be the first push
                logger.warning("Pull failed, might be first push")
            
            # Push changes
            logger.info(f"Pushing to {branch}...")
            subprocess.run(
                ['git', 'push', '-u', 'origin', branch],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                timeout=30
            )
            
            logger.info(f"✅ Pushed changes to {branch}")
            return True, None
            
        except subprocess.TimeoutExpired:
            error_msg = "Push operation timed out"
            logger.error(error_msg)
            return False, error_msg
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Failed to push: {error_msg}")
            return False, error_msg
    
    def create_branch(self, branch_name: str) -> bool:
        """
        Create and checkout a new branch.
        
        Args:
            branch_name: Name of the branch to create
            
        Returns:
            True if successful
        """
        try:
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            logger.info(f"✅ Created and checked out branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    
    def get_git_status(self) -> Dict:
        """
        Get current git repository status.
        
        Returns:
            Dict with status information
        """
        try:
            # Get current branch
            branch_result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            current_branch = branch_result.stdout.strip()
            
            # Get status
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            # Get last commit
            log_result = subprocess.run(
                ['git', 'log', '-1', '--oneline'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            last_commit = log_result.stdout.strip()
            
            # Count commits
            count_result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            total_commits = int(count_result.stdout.strip())
            
            return {
                'branch': current_branch,
                'status': status_result.stdout,
                'last_commit': last_commit,
                'total_commits': total_commits,
                'has_changes': bool(status_result.stdout.strip())
            }
        except Exception as e:
            logger.error(f"Failed to get git status: {e}")
            return {
                'error': str(e)
            }
    
    def commit_and_push_evolution(self,
                                  version: int,
                                  files_changed: List[str],
                                  metrics: Dict,
                                  description: str,
                                  branch: str = 'main') -> Dict:
        """
        Complete workflow: commit and push code evolution.
        
        Returns:
            Dict with results
        """
        logger.info(f"🔄 Starting autonomous commit workflow for v{version}...")
        
        result = {
            'version': version,
            'timestamp': datetime.utcnow().isoformat(),
            'commit_success': False,
            'push_success': False,
            'commit_hash': None,
            'error': None
        }
        
        # Commit
        commit_success, commit_hash = self.commit_code_evolution(
            version, files_changed, metrics, description
        )
        
        result['commit_success'] = commit_success
        result['commit_hash'] = commit_hash
        
        if not commit_success:
            result['error'] = 'Commit failed'
            return result
        
        # Push
        push_success, push_error = self.push_changes(branch)
        result['push_success'] = push_success
        
        if not push_success:
            result['error'] = f'Push failed: {push_error}'
            return result
        
        logger.info(f"✅ Evolution v{version} committed and pushed successfully!")
        return result


def test_github_integration():
    """Test GitHub integration."""
    github = GitHubIntegration()
    
    print("Testing GitHub Integration...")
    print("="*60)
    
    # Test 1: Initialize
    print("\n1. Initializing repository...")
    if github.initialize_repo():
        print("✅ Initialization successful")
    else:
        print("❌ Initialization failed")
        return
    
    # Test 2: Get status
    print("\n2. Getting repository status...")
    status = github.get_git_status()
    print(f"Branch: {status.get('branch')}")
    print(f"Last commit: {status.get('last_commit')}")
    print(f"Total commits: {status.get('total_commits')}")
    print(f"Has changes: {status.get('has_changes')}")
    
    # Test 3: Test commit (if there are changes)
    if status.get('has_changes'):
        print("\n3. Testing commit...")
        success, commit_hash = github.commit_code_evolution(
            version=999,
            files_changed=['README.md'],
            metrics={'test': True},
            description='Test commit from GitHub integration test'
        )
        if success:
            print(f"✅ Commit successful: {commit_hash[:8]}")
        else:
            print("❌ Commit failed")
    else:
        print("\n3. No changes to commit")
    
    print("\n" + "="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_github_integration()

