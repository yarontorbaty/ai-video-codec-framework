#!/usr/bin/env python3
"""Test GitHub integration"""
import sys
import os
sys.path.insert(0, '/home/ec2-user/ai-video-codec')

from src.agents.github_integration import GitHubIntegration

print("="*60)
print("Testing GitHub Integration")
print("="*60)

gh = GitHubIntegration()

print("\n1. Initializing repository...")
if gh.initialize_repo():
    print("✅ Repository initialized")
    
    # Create a test file
    test_file = "src/agents/test_evolution.py"
    test_path = os.path.join(gh.repo_path, test_file)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    with open(test_path, "w") as f:
        f.write("# Test file for GitHub integration\n")
        f.write("# Created by autonomous system test\n")
        f.write("# This verifies the LLM can commit to GitHub\n")
    
    print(f"✅ Created test file: {test_file}")
    
    # Test commit
    print("\n2. Creating test commit...")
    success, commit_hash = gh.commit_code_evolution(
        version=0,
        files_changed=[test_file],
        metrics={"test": True, "bitrate_mbps": 1.5},
        description="Testing GitHub integration - LLM autonomous commit system"
    )
    
    if success:
        print(f"✅ Commit successful: {commit_hash[:8] if commit_hash else 'unknown'}")
        
        # Test push
        print("\n3. Pushing to GitHub...")
        push_success, push_error = gh.push_changes()
        
        if push_success:
            print("✅ Push successful!")
            print("\n" + "="*60)
            print("SUCCESS! Check your GitHub repository!")
            print("="*60)
        else:
            print(f"❌ Push failed: {push_error}")
    else:
        print("❌ Commit failed")
else:
    print("❌ Repository initialization failed")

