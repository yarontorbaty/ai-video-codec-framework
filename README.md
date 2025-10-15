# AI Video Codec Framework

> **Autonomous AI-based video codec development system achieving 90%+ bitrate reduction vs HEVC with 95%+ PSNR retention**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen.svg)](LICENSE_ANALYSIS.md)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![AWS](https://img.shields.io/badge/cloud-AWS%20%7C%20GCP%20%7C%20Azure-orange.svg)](docs/)

---

## 🎯 Project Goals

This framework autonomously develops next-generation AI-based video codecs through continuous experimentation and optimization.

**Target Performance:**
- 📉 **90%+ bitrate reduction** compared to HEVC
- 📊 **95%+ PSNR** quality retention
- ⚡ **Real-time 4K60** encode/decode on 40 TOPS hardware
- 💰 **<$5,000/month** AWS operational costs

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Autonomous Orchestrator                      │
│  • Experiment planning & execution                           │
│  • Meta-learning from results                                │
│  • Cost tracking & optimization                              │
│  • Hourly progress reporting                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┬──────────────┐
     │             │             │              │
┌────▼─────┐ ┌────▼─────┐ ┌─────▼────┐ ┌──────▼──────┐
│ Training │ │ Inference│ │Evaluation│ │   Reports   │
│  (GPU)   │ │  (GPU)   │ │  (CPU)   │ │   & Logs    │
└──────────┘ └──────────┘ └──────────┘ └─────────────┘
```

### Hybrid Compression Approach

1. **Semantic Extraction** - Motion, scene understanding, temporal analysis
2. **Neural Encoding** - Learned compression to compact latent space
3. **Generative Reconstruction** - High-quality frame synthesis from latents

## 📚 Documentation

- **[AI_VIDEO_CODEC_FRAMEWORK.md](AI_VIDEO_CODEC_FRAMEWORK.md)** - Comprehensive framework overview
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Technical implementation details
- **[TIMELINE_AND_MILESTONES.md](TIMELINE_AND_MILESTONES.md)** - 7-day sprint plan and milestones

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- AWS account with appropriate permissions
- 4K60 test video (10 seconds) + HEVC reference

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/ai-video-codec.git
cd ai-video-codec

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start local orchestrator (for testing)
python orchestrator/master.py --config config/local_config.yaml
```

### AWS Deployment

```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure
bash scripts/deploy_framework.sh

# Monitor progress
tail -f /var/log/orchestrator.log

# Or use monitoring script
python scripts/monitor.py --follow
```

## 📈 Timeline

### Phase 1: Framework Development (Days 1-2)
- ✅ AWS infrastructure setup
- ✅ Orchestrator implementation
- ✅ Training/evaluation pipeline
- ✅ Monitoring and reporting

### Phase 2: Alpha Codec (Days 3-7)
- 🔄 Baseline model training
- 🔄 Architecture exploration
- 🔄 Hybrid approach implementation
- 🔄 Performance optimization
- 🎯 **Alpha Release: Day 7**

### Phase 3: Beta (Days 8-14)
- 🔜 Real-time optimization
- 🔜 Hardware acceleration
- 🔜 Robustness testing
- 🎯 **Beta Release: Day 14**

## 📊 Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| **M1:** Framework Operational | Day 2 | 🟡 In Progress |
| **M2:** Proof of Concept | Day 4 | ⚪ Pending |
| **M3:** Compression Target (90%) | Day 5 | ⚪ Pending |
| **M4:** Quality Target (PSNR 95%) | Day 6 | ⚪ Pending |
| **M5:** Alpha Release | Day 7 | ⚪ Pending |
| **M6:** Production Ready | Day 14 | ⚪ Pending |

## 🧪 Codec Models

The framework explores multiple compression approaches:

### Baseline Models
- **Simple Autoencoder** - Sanity check and baseline
- **Scale Hyperprior** - Proven neural compression (Ballé et al.)
- **VQ-VAE** - Vector quantized compression

### Advanced Models
- **Hybrid Semantic** - Keyframes + motion + generative refinement
- **Generative** - Diffusion/GAN-based reconstruction
- **Custom Architectures** - Evolved through experimentation

## 📉 Optimization Techniques

- **Model Compression:** Pruning, quantization (INT8/INT4)
- **Knowledge Distillation:** Large teacher → small student
- **Architecture Search:** NAS for efficiency
- **Hardware Optimization:** TensorRT, operator fusion

## 💰 Cost Management

**Budget:** $5,000/month maximum

**Strategy:**
- Spot instances (70% savings on training)
- Auto-scaling based on experiment queue
- Aggressive cost monitoring and alerts
- Automatic shutdown at 95% budget

**Projected Costs:**
- Week 1 (Alpha): $960-1,400
- Week 2 (Beta): $700-1,100
- **Total:** ~$2,500 (50% of budget)

## 📝 Experiment Tracking

All experiments logged with:
- Architecture configuration
- Hyperparameters
- Training metrics
- Quality metrics (PSNR, SSIM, VMAF)
- Compression ratio
- Inference speed
- Model size
- Cost

Access experiment database:
```python
from orchestrator.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
best = tracker.get_best_experiments(metric='psnr', top_k=10)
```

## 📊 Monitoring & Reporting

**Hourly Reports Include:**
- Current training status
- Best results so far
- Experiments completed
- Cost tracking (hourly, daily, monthly)
- Next planned actions

**Access Reports:**
```bash
# View latest report
cat /var/log/hourly_reports/latest.txt

# Or via S3
aws s3 sync s3://ai-video-codec-reports ./reports/
```

## 🧰 Tools & Scripts

```bash
# Run single experiment
python scripts/run_experiment.py --config experiments/experiment_27.yaml

# Evaluate codec
python scripts/evaluate_codec.py \
  --model models/best_model.pth \
  --video data/test_4k60.mp4

# Optimize model for deployment
python scripts/optimize_model.py \
  --input models/best_model.pth \
  --output models/optimized_int8.onnx \
  --quantize int8

# Generate comparison report
python scripts/compare_codecs.py \
  --codec1 hevc \
  --codec2 ai_codec \
  --video data/test_4k60.mp4 \
  --output comparison_report.html
```

## 🔬 Research & References

**Key Papers:**
1. Ballé et al. (2018) - "Variational Image Compression with a Scale Hyperprior"
2. Mentzer et al. (2020) - "High-Fidelity Generative Image Compression"
3. Yang et al. (2021) - "Conditional Variational Autoencoder for Neural Video Compression"

**Libraries Used:**
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) - Neural compression
- [PyTorch](https://pytorch.org/) - Deep learning
- [FFmpeg](https://ffmpeg.org/) - Video processing

## 🤝 Contributing

This is currently a research project. Contributions welcome after initial alpha release.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

**Apache License 2.0** - See [LICENSE](LICENSE) file for details

This project is fully open source and compatible with commercial use.

### Why Apache 2.0?
- ✅ Permissive license allowing commercial use
- ✅ Explicit patent protection
- ✅ Compatible with all dependencies
- ✅ Industry standard for ML projects (PyTorch, TensorFlow use it)

### Third-Party Licenses
All dependencies are permissively licensed (Apache 2.0, MIT, BSD).

See [LICENSE_ANALYSIS.md](LICENSE_ANALYSIS.md) for detailed dependency license information.

### Optional Dependencies
Some optional features (TensorRT, W&B) have separate licenses:
- **TensorRT**: Proprietary (NVIDIA), freely redistributable, OPTIONAL
- **W&B**: MIT client + proprietary service, OPTIONAL (use TensorBoard instead)

See [requirements-optional.txt](requirements-optional.txt) for full list.

## 🎯 Current Status

**Project Start:** October 16, 2025  
**Current Phase:** Infrastructure Setup (Day 1)  
**Next Milestone:** Framework Operational (Day 2)

**Latest Results:**
- Experiments Run: 0
- Best Compression: N/A
- Best PSNR: N/A
- Cost This Month: $0.00

## 📞 Contact & Support

For questions or issues:
- Create an issue on GitHub
- Check the [FAQ](docs/FAQ.md)
- Review hourly progress reports

---

**Built with ❤️ for the future of video compression**

