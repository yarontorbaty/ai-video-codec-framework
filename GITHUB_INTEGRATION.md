# GitHub Integration for LLM Autonomous Commits

## Overview

The LLM can now **autonomously commit and push code improvements** to GitHub. All credentials are stored securely in AWS Secrets Manager (never in the repository).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│         LLM Successfully Evolves Code (v2 → v3)           │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│           AdaptiveCodecAgent.save_implementation()        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  1. Save to /tmp/best_codec_implementation.json    │  │
│  │  2. Write code to src/agents/evolved_codec.py      │  │
│  │  3. Call GitHubIntegration.commit_and_push()       │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              GitHub Integration Module                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │  1. Get credentials from AWS Secrets Manager       │  │
│  │  2. Configure git (user, email, remote)            │  │
│  │  3. Stage changed files                            │  │
│  │  4. Create commit with performance metrics         │  │
│  │  5. Push to GitHub                                 │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                    GitHub Repository                      │
│  📝 Commit: "🤖 Autonomous Code Evolution - v3"          │
│     LLM evolved codec to v3 - 2.1 Mbps, 3.5x compression│
│     Performance Metrics:                                  │
│       • Bitrate: 2.10 Mbps                               │
│       • Compression: 3.50x                               │
│       • Improvement: 34% better than v2                  │
└──────────────────────────────────────────────────────────┘
```

## Setup

### 1. Create GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "AI Video Codec LLM Commits"
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you'll need it in next step)

### 2. Run Setup Script

```bash
chmod +x scripts/setup_github_credentials.sh
./scripts/setup_github_credentials.sh
```

The script will prompt for:
- GitHub username
- GitHub personal access token (from step 1)
- GitHub email (for commits)
- Repository URL (e.g., `https://github.com/yourusername/AiV1.git`)

### 3. Verify Setup

Check that credentials are stored:
```bash
aws secretsmanager get-secret-value \
  --secret-id ai-video-codec/github-credentials \
  --region us-east-1 \
  --query SecretString \
  --output text | jq .
```

Should show (token partially redacted):
```json
{
  "username": "yourusername",
  "token": "ghp_xxxxxxxxxxxx",
  "email": "your@email.com",
  "repo": "https://github.com/yourusername/AiV1.git"
}
```

## How It Works

### Automatic Commits

When the LLM successfully evolves the codec:

1. **Code Evolution Detected**
   ```python
   # In adaptive_codec_agent.py
   if should_adopt:
       self.save_implementation(code, metrics)
       # ↓ Automatically commits to GitHub
   ```

2. **File Saved**
   - Code saved to `/tmp/best_codec_implementation.json`
   - Code written to `src/agents/evolved_codec.py`

3. **Git Commit Created**
   ```
   🤖 Autonomous Code Evolution - v3
   
   LLM evolved codec to v3 - 2.1 Mbps, 3.5x compression
   
   Performance Metrics:
     • Bitrate: 2.10 Mbps
     • Compression: 3.50x
     • Improvement: 34% better than v2
   
   Timestamp: 2025-10-16T23:45:12.345678
   Evolved by: LLM Autonomous System
   ```

4. **Pushed to GitHub**
   - Commit pushed to `main` branch
   - Visible in commit history
   - Full metrics included

### Manual Commits (via LLM Chat)

The LLM can also commit manually via chat:

```
User: "Commit the latest improvements to GitHub"

LLM: [Calls GitHubIntegration.commit_and_push_evolution()]
     
     Response: "✅ Committed v3 to GitHub
               Commit: abc123de
               Files: src/agents/evolved_codec.py
               Metrics: 2.1 Mbps, 3.5x compression
               
               View at: https://github.com/user/AiV1/commit/abc123de"
```

## GitHubIntegration API

### Core Methods

#### `commit_code_evolution(version, files_changed, metrics, description)`

Creates a git commit with detailed metrics.

```python
from agents.github_integration import GitHubIntegration

github = GitHubIntegration()

success, commit_hash = github.commit_code_evolution(
    version=3,
    files_changed=['src/agents/evolved_codec.py'],
    metrics={
        'bitrate_mbps': 2.1,
        'compression_ratio': 3.5,
        'improvement': '34% better than v2'
    },
    description='Improved JPEG quality settings'
)

# success: True
# commit_hash: 'abc123def456...'
```

#### `push_changes(branch='main')`

Pushes committed changes to remote.

```python
success, error = github.push_changes(branch='main')

# success: True
# error: None
```

#### `commit_and_push_evolution(...)` 

Complete workflow in one call.

```python
result = github.commit_and_push_evolution(
    version=3,
    files_changed=['src/agents/evolved_codec.py'],
    metrics={'bitrate_mbps': 2.1},
    description='Improved codec',
    branch='main'
)

# result: {
#   'version': 3,
#   'timestamp': '2025-10-16T23:45:12',
#   'commit_success': True,
#   'push_success': True,
#   'commit_hash': 'abc123...',
#   'error': None
# }
```

#### `get_git_status()`

Get current repository status.

```python
status = github.get_git_status()

# status: {
#   'branch': 'main',
#   'status': ' M src/agents/evolved_codec.py\n',
#   'last_commit': 'abc123d Autonomous Code Evolution - v3',
#   'total_commits': 42,
#   'has_changes': True
# }
```

## Commit Message Format

### Standard Format

```
🤖 Autonomous Code Evolution - v{version}

{description}

Performance Metrics:
  • Bitrate: {X.XX} Mbps
  • Compression: {X.XX}x
  • Improvement: {improvement_description}

Timestamp: {ISO8601_timestamp}
Evolved by: LLM Autonomous System
```

### Example

```
🤖 Autonomous Code Evolution - v5

LLM evolved codec to v5 - 1.8 Mbps, 4.2x compression

Performance Metrics:
  • Bitrate: 1.80 Mbps
  • Compression: 4.20x
  • Improvement: bitrate_reduction_percent: 15.3

Timestamp: 2025-10-16T23:45:12.345678
Evolved by: LLM Autonomous System
```

## Security

### Credentials Storage

- ✅ Stored in AWS Secrets Manager
- ✅ Encrypted at rest
- ✅ Access via IAM roles only
- ✅ Never in repository
- ✅ Never in logs

### IAM Permissions

Orchestrator instance needs:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:*:*:secret:ai-video-codec/github-credentials-*"
    }
  ]
}
```

### Git Authentication

- Uses HTTPS with token authentication
- Token embedded in remote URL temporarily
- Never written to disk permanently
- Cleared after each operation

## Monitoring

### Via Dashboard

The dashboard will show (when dashboard view is implemented):
- Latest commit hash
- Commits in last 24 hours
- Current branch
- Files changed

### Via Logs

Check orchestrator logs for GitHub activity:
```bash
aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["grep -i \"github\\|commit\\|push\" /tmp/orch.log | tail -20"]'
```

### Via Admin Chat

```
User: "Show GitHub status"

LLM: [Calls get_git_status()]
     
     Response: "Current status:
               Branch: main
               Last commit: abc123d (v5)
               Total commits: 42
               Uncommitted changes: No
               
               Latest evolutions:
               • v5: 1.8 Mbps (1 hour ago)
               • v4: 2.0 Mbps (3 hours ago)
               • v3: 2.1 Mbps (5 hours ago)"
```

## Workflow Examples

### Scenario 1: Successful Evolution

```
1. LLM generates improved compression code
2. Code passes validation ✅
3. Code passes execution tests ✅
4. Performance: 1.8 Mbps (better than 2.1 Mbps) ✅
5. Code adopted as v5
6. ↓ Automatic GitHub commit
7. File: src/agents/evolved_codec.py
8. Commit: "🤖 Autonomous Code Evolution - v5..."
9. Push to main branch ✅
10. Visible on GitHub!
```

### Scenario 2: Multiple Evolutions

```
v1 → v2 (commit: abc1234)
      ↓ 30 minutes later
v2 → v3 (commit: def5678)
      ↓ 1 hour later
v3 → v4 (commit: ghi9012)

Git history shows:
  ghi9012 - 🤖 Autonomous Code Evolution - v4 (1 hour ago)
  def5678 - 🤖 Autonomous Code Evolution - v3 (1.5 hours ago)
  abc1234 - 🤖 Autonomous Code Evolution - v2 (2 hours ago)
```

### Scenario 3: LLM-Triggered Commit

```
User: "The codec has improved significantly. Commit it."

LLM: [Checks if there are uncommitted changes]
     [Calls commit_and_push_evolution()]
     
     "✅ Committed v6 to GitHub
     
      Details:
      • Commit: xyz7890
      • Bitrate: 1.5 Mbps
      • Compression: 5.1x
      • Files: src/agents/evolved_codec.py
      
      This is a 16% improvement over v5!
      
      View commit: https://github.com/user/AiV1/commit/xyz7890"
```

## Files Committed

### Primary Files

1. **`src/agents/evolved_codec.py`**
   - The actual evolved compression code
   - Contains the LLM-generated function
   - Updated each evolution

### Future Files (as system expands)

- `src/agents/evolved_codec_v{N}.py` - Version history
- `docs/evolution_log.md` - Human-readable change log
- `metrics/evolution_metrics.json` - Performance tracking

## Benefits

### For LLM

- ✅ Full autonomy over code evolution
- ✅ Automatic version control
- ✅ Git history of all improvements
- ✅ Easy rollback to previous versions
- ✅ Transparent progress tracking

### For Humans

- ✅ See every improvement in GitHub
- ✅ Review commit history
- ✅ Understand what changed
- ✅ Metrics in commit messages
- ✅ Easy to audit and review

### For System

- ✅ Persistent evolution record
- ✅ Disaster recovery (git history)
- ✅ Collaboration ready (multiple LLMs)
- ✅ Integration with CI/CD (future)

## Troubleshooting

### "Failed to commit: authentication required"

Check credentials:
```bash
aws secretsmanager get-secret-value \
  --secret-id ai-video-codec/github-credentials \
  --region us-east-1
```

Verify token hasn't expired on GitHub.

### "Failed to push: repository not found"

Check repository URL in secrets:
```bash
aws secretsmanager get-secret-value \
  --secret-id ai-video-codec/github-credentials \
  --region us-east-1 \
  --query SecretString \
  --output text | jq -r .repo
```

### "Failed to push: insufficient permissions"

Ensure GitHub token has `repo` scope:
1. Go to: https://github.com/settings/tokens
2. Click your token
3. Verify `repo` is checked
4. Regenerate if needed
5. Update secret

## Future Enhancements

1. **Pull Requests**: LLM creates PRs instead of direct commits
2. **Branching**: Each evolution on separate branch
3. **Code Review**: LLM reviews its own PRs
4. **Changelog**: Automatic CHANGELOG.md generation
5. **Tagging**: Git tags for major versions
6. **CI/CD**: Automatic testing on commit
7. **Multi-LLM**: Collaboration via git

## Conclusion

The LLM now has **complete control over its code evolution** with full Git integration:

- ✅ Autonomous commits
- ✅ Secure credential management
- ✅ Detailed commit messages
- ✅ Full version history
- ✅ GitHub visibility

This enables true **autonomous research** with complete transparency and reproducibility!

