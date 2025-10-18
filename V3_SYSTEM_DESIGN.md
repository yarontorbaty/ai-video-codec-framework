# AI Video Codec Framework - v3.0 System Design

## Executive Summary

This document outlines the complete redesign of the AI Video Codec Framework (v3.0) - a system that uses LLM-powered agents to evolve video compression algorithms through iterative experimentation.

### Core Goals
1. **Autonomous Evolution**: LLM generates, tests, and improves video compression code
2. **Real Metrics**: Measure actual PSNR, SSIM, bitrate, compression ratio
3. **Reproducibility**: Save decoder code and reconstructed videos for each experiment
4. **Observability**: Real-time dashboards showing progress and metrics
5. **Reliability**: Simple, tested architecture that actually works

---

## System Architecture (v3.0)

### Design Principles
1. **Simplicity First**: No complex microservices, minimal moving parts
2. **Test Everything**: Unit tests before deployment
3. **Observable**: Comprehensive logging and metrics
4. **Fail Fast**: Clear error messages, quick debugging
5. **Incremental**: Build and test one component at a time

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     CloudFront (CDN)                        │
│                 dashboard.example.com                       │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼────────┐    ┌────────▼────────┐
│  Lambda (SSR)   │    │  Lambda (API)   │
│  Dashboard HTML │    │  Admin Actions  │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │    DynamoDB         │
         │  - experiments      │
         │  - metrics          │
         │  - evolution_log    │
         └──────────┬──────────┘
                    │
    ┌───────────────┴───────────────┐
    │                               │
┌───▼────────────┐      ┌──────────▼───────┐
│  S3 Buckets    │      │  Secrets Manager │
│  - videos      │      │  - anthropic_key │
│  - decoders    │      └──────────────────┘
│  - artifacts   │
└────────────────┘

┌──────────────────────────────────────────────────────────┐
│              EC2 Orchestrator (t3.medium)                │
│  - Runs autonomous_orchestrator.py                       │
│  - Calls Anthropic Claude for code generation            │
│  - Manages experiment lifecycle                          │
│  - SSM enabled for remote management                     │
└────────────┬─────────────────────────────────────────────┘
             │
             │ (Internal network)
             │
┌────────────▼─────────────────────────────────────────────┐
│           EC2 GPU Worker (g4dn.xlarge)                   │
│  - Executes video compression experiments                │
│  - Calculates PSNR/SSIM metrics                          │
│  - Uploads results to S3 and DynamoDB                    │
│  - SSM enabled for remote management                     │
└──────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Orchestrator (EC2 Instance)

**Purpose**: Coordinate experiments and LLM interactions

**Responsibilities**:
- Generate compression code using Claude API
- Create experiment definitions
- Send experiments to GPU worker
- Monitor experiment status
- Store results in DynamoDB
- Manage evolution history

**Tech Stack**:
- Python 3.10+
- anthropic SDK
- boto3 (AWS SDK)
- Simple HTTP server (no Flask complexity)

**Key Files**:
- `orchestrator/main.py` - Main loop
- `orchestrator/llm_client.py` - Claude API wrapper
- `orchestrator/experiment_manager.py` - Experiment lifecycle
- `orchestrator/config.py` - Configuration

### 2. GPU Worker (EC2 Instance)

**Purpose**: Execute video compression experiments with real metrics

**Responsibilities**:
- Receive experiment from orchestrator
- Execute encoding code on test video
- Execute decoding code to reconstruct video
- Calculate PSNR, SSIM, bitrate, compression ratio
- Upload reconstructed video to S3
- Save decoder code to S3
- Store metrics in DynamoDB

**Tech Stack**:
- Python 3.10+
- OpenCV (cv2)
- scikit-image (for SSIM)
- numpy
- boto3
- Simple HTTP server

**Key Files**:
- `worker/main.py` - HTTP server
- `worker/experiment_runner.py` - Execute experiments
- `worker/metrics.py` - Calculate PSNR/SSIM
- `worker/s3_uploader.py` - Upload artifacts

### 3. Lambda Functions

#### Dashboard SSR (`lambda/dashboard_ssr.py`)
- Server-side render HTML dashboard
- Query DynamoDB for latest experiments
- Display metrics, status, evolution progress
- Cache-Control: 30 seconds

#### Admin API (`lambda/admin_api.py`)
- Rerun experiments
- Stop/start orchestrator
- View detailed experiment data
- Export results

### 4. DynamoDB Tables

#### `experiments` Table
```json
{
  "experiment_id": "exp_1234567890",  // PK
  "timestamp": 1234567890,             // SK (sort key)
  "status": "completed",               // running|completed|failed
  "phase": "evolution",                // baseline|evolution|analysis
  "
  "metrics": {
    "psnr_db": 34.52,
    "ssim": 0.95,
    "bitrate_mbps": 2.1,
    "compression_ratio": 42.5,
    "original_size_bytes": 10485760,
    "compressed_size_bytes": 248832
  },
  "artifacts": {
    "video_url": "https://s3.../reconstructed_video.mp4",
    "decoder_s3_key": "decoders/decoder_1234567890.py",
    "encoder_s3_key": "encoders/encoder_1234567890.py"
  },
  "evolution": {
    "iteration": 5,
    "parent_experiment_id": "exp_1234567889",
    "improvement_psnr": 2.3,
    "llm_reasoning": "Improved motion compensation..."
  },
  "runtime_seconds": 45.2,
  "test_count": 10
}
```

---

## Data Schemas

### Experiment Lifecycle States
1. **queued** - Waiting to execute
2. **running** - Currently executing
3. **completed** - Successfully finished
4. **failed** - Execution error
5. **analyzed** - LLM has analyzed results

### Metrics Schema
```python
{
    "psnr_db": float,          # Peak Signal-to-Noise Ratio (higher is better, >30 is good)
    "ssim": float,             # Structural Similarity (0-1, >0.9 is good)
    "bitrate_mbps": float,     # Megabits per second
    "compression_ratio": float, # Original size / Compressed size
    "original_size_bytes": int,
    "compressed_size_bytes": int,
    "processing_time_ms": int
}
```

---

## Control Mechanisms

### 1. Experiment Control
- **Start Orchestrator**: SSM command or Lambda API
- **Stop Orchestrator**: Graceful shutdown via signal
- **Pause Experiments**: Set DynamoDB flag
- **Emergency Stop**: Kill process via SSM

### 2. Rate Limiting
- Max 1 experiment per minute (GPU cooling)
- Max 100 experiments per day (cost control)
- LLM API: 50 requests per minute

### 3. Resource Management
- Orchestrator: 2GB RAM, 2 vCPUs
- GPU Worker: 16GB RAM, 4 vCPUs, 16GB GPU
- S3: Lifecycle policy to delete old artifacts after 30 days
- DynamoDB: On-demand pricing

---

## Observability

### 1. Logging
**Orchestrator Logs** (`/var/log/orchestrator.log`):
- LLM requests/responses
- Experiment submissions
- Error traces

**Worker Logs** (`/var/log/worker.log`):
- Experiment execution
- Metric calculations
- S3 uploads
- Errors

**CloudWatch Integration**:
- All logs forwarded to CloudWatch
- Retention: 7 days
- Log groups: `/ai-codec/orchestrator`, `/ai-codec/worker`

### 2. Metrics (CloudWatch)
- `ExperimentsCompleted` - Count per hour
- `ExperimentsFailed` - Count per hour
- `AveragePSNR` - Average across experiments
- `AverageSSIM` - Average across experiments
- `LLMAPILatency` - p50, p95, p99
- `WorkerUtilization` - CPU, Memory, GPU

### 3. Dashboards

#### Public Dashboard
- Latest 20 experiments with metrics
- Evolution progress chart (PSNR over iterations)
- Current best compression ratio
- System status (running/stopped)
- Links to reconstructed videos

#### Admin Dashboard
- All experiments (paginated)
- Detailed metrics
- Rerun button
- Stop/start controls
- LLM reasoning viewer
- Error logs
- Resource usage charts

---

## Testing Strategy

### 1. Unit Tests
```
tests/
├── test_orchestrator.py
├── test_worker.py
├── test_metrics.py
├── test_llm_client.py
└── test_s3_uploader.py
```

**Coverage Target**: >80%

**Test Categories**:
- Metrics calculation accuracy
- Code execution sandbox
- S3 upload/download
- DynamoDB operations
- LLM prompt generation

### 2. Integration Tests
- Full experiment end-to-end
- Orchestrator → Worker → DynamoDB → S3
- Dashboard rendering with real data
- Admin API operations

### 3. Validation Tests
- Verify decoded video plays correctly
- Verify PSNR/SSIM calculations match reference
- Verify S3 files are accessible
- Verify DynamoDB data integrity

---

## Security

### 1. Authentication & Authorization
- **API Keys**: Stored in AWS Secrets Manager
- **IAM Roles**: Least privilege per component
- **Network**: GPU worker in private subnet
- **Dashboard**: CloudFront with WAF (optional)

### 2. Code Execution Safety
- **Sandbox**: Execute LLM code in restricted environment
- **Timeouts**: Kill runaway processes after 60s
- **Resource Limits**: Memory and CPU caps
- **AST Validation**: Check code before execution
- **Whitelist**: Only allow safe imports

### 3. Data Protection
- **S3 Encryption**: AES-256 at rest
- **DynamoDB Encryption**: Default encryption
- **Secrets**: Never log API keys
- **Videos**: Presigned URLs with 1-hour expiration

---

## Deployment Process

### Phase 1: Infrastructure (CloudFormation)
```bash
# Create VPC, subnets, security groups
aws cloudformation create-stack --stack-name ai-codec-network \
  --template-body file://infra/network.yaml

# Create DynamoDB tables
aws cloudformation create-stack --stack-name ai-codec-database \
  --template-body file://infra/database.yaml

# Create S3 buckets
aws cloudformation create-stack --stack-name ai-codec-storage \
  --template-body file://infra/storage.yaml

# Create Lambda functions
aws cloudformation create-stack --stack-name ai-codec-lambda \
  --template-body file://infra/lambda.yaml

# Create EC2 instances with SSM
aws cloudformation create-stack --stack-name ai-codec-compute \
  --template-body file://infra/compute.yaml
```

### Phase 2: Application Code
```bash
# Deploy orchestrator
./deploy/deploy_orchestrator.sh

# Deploy worker
./deploy/deploy_worker.sh

# Deploy Lambda functions
./deploy/deploy_lambda.sh
```

### Phase 3: Validation
```bash
# Run integration tests
python tests/integration_test.py

# Run first experiment
python scripts/run_first_experiment.py

# Verify dashboard
curl https://dashboard.example.com
```

---

## Success Criteria

### Must Have (v3.0 MVP)
- ✅ 2 successful experiments with real PSNR/SSIM
- ✅ Videos uploaded and playable
- ✅ Decoder code saved and downloadable
- ✅ Dashboard shows metrics
- ✅ LLM generates working compression code
- ✅ All components deployed and running

### Nice to Have
- 🎯 10+ iterations of evolution
- 🎯 PSNR improvement over baseline
- 🎯 Compression ratio >50
- 🎯 Evolution visualization chart
- 🎯 Admin dashboard with controls

### Future Enhancements (v3.1+)
- Multi-GPU parallelization
- Real-time streaming experiments
- A/B testing different LLM models
- Community leaderboard
- Neural network integration
- Multi-codec comparison

---

## Lessons Learned from v2.0

### What Went Wrong
1. **Over-complexity**: Too many services, hard to debug
2. **Deployment Hell**: Python caching issues, file sync problems
3. **No Incremental Testing**: Built everything, tested nothing
4. **Mixed Concerns**: Worker doing too many things
5. **Poor Observability**: Couldn't see what was happening

### What We're Changing
1. **Simplicity**: 2 EC2 instances, that's it
2. **Test First**: Unit tests before deployment
3. **Clear Interfaces**: Simple HTTP between components
4. **One Thing Well**: Each component has one job
5. **Logging Everything**: Know exactly what's happening

---

## Development Timeline

### Day 1 (Tonight - 8 hours)
- ✅ Document design (this file)
- 🔨 Create v3.0 branch
- 🔨 Nuke old AWS resources
- 🔨 Deploy new infrastructure
- 🔨 Build orchestrator (basic)
- 🔨 Build worker (basic)
- 🔨 Deploy and test end-to-end
- 🔨 Run 2 successful experiments
- 🔨 Verify dashboard works

### Day 2-3 (Polish)
- Add unit tests
- Improve LLM prompts
- Add evolution logic
- Enhance dashboards
- Documentation

### Day 4+ (Iterate)
- Monitor real experiments
- Fix issues
- Optimize performance
- Add features

---

## Monitoring & Alerts

### Critical Alerts (PagerDuty/SNS)
- Orchestrator down >5 minutes
- Worker crashed
- DynamoDB throttling
- S3 upload failures
- >5 failed experiments in a row

### Warning Alerts (Email)
- High GPU utilization (>90% for >30min)
- Slow LLM API (>10s latency)
- Low PSNR scores (<25 dB)
- Disk space low (<20%)

### Info Notifications (Slack)
- New best PSNR achieved
- 10 experiments completed
- System restarted
- Evolution milestone reached

---

## Cost Estimates

### Monthly AWS Costs (v3.0)
- EC2 Orchestrator (t3.medium, 24/7): ~$30
- EC2 GPU Worker (g4dn.xlarge, 8h/day): ~$200
- DynamoDB (on-demand, 10K writes): ~$1
- S3 (1TB storage, 10K downloads): ~$25
- Lambda (100K invocations): ~$0.20
- CloudFront (10GB transfer): ~$1
- CloudWatch (logs + metrics): ~$10
- **Total: ~$267/month**

### Cost Optimization
- Spot instances for GPU (-70%)
- S3 lifecycle policies (move to Glacier after 7 days)
- DynamoDB reserved capacity
- CloudFront caching
- **Optimized Total: ~$100/month**

---

## FAQ

**Q: Why not use SageMaker or Lambda for GPU?**
A: Too expensive, too complex. Simple EC2 gives us full control.

**Q: Why Python instead of Go/Rust?**
A: ML ecosystem is Python. OpenCV, scikit-image, anthropic SDK.

**Q: Why not Kubernetes?**
A: Overkill. 2 instances don't need orchestration.

**Q: How do we scale to 100 experiments/hour?**
A: Add more GPU workers behind a load balancer. But start simple.

**Q: What if the LLM generates malicious code?**
A: Sandbox execution with AST validation and resource limits.

---

## Conclusion

v3.0 is a clean slate. We've learned what doesn't work (v2.0 complexity) and we're building something that will actually work:

1. **Simple** - 2 instances, clear responsibilities
2. **Testable** - Unit tests before deployment
3. **Observable** - Know exactly what's happening
4. **Reliable** - No deployment mysteries
5. **Functional** - Real metrics, real videos, real progress

Let's build it right this time. 🚀

