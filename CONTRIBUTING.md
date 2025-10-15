# Contributing to AI Video Codec Framework

First off, thank you for considering contributing to the AI Video Codec Framework! It's people like you that make this project such a great tool for the research and video compression community.

## 🎯 Project Status

This project is currently in **early development** (pre-alpha). We're building the autonomous framework and initial codec implementations. Contributions are welcome, but please understand that the project structure and APIs may change significantly during this phase.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- CUDA 11.8+ (for GPU support)
- Familiarity with PyTorch and video processing

### Setting Up Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AiV1.git
cd AiV1

# Add upstream remote
git remote add upstream https://github.com/yarontorbaty/AiV1.git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear descriptive title**
- **Detailed description** of the issue
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, GPU, etc.)
- **Error messages or logs**
- **Screenshots** if applicable

Use the bug report template when creating an issue.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear descriptive title**
- **Detailed description** of the proposed feature
- **Use cases** and benefits
- **Possible implementation** approach (optional)
- **Alternatives considered** (optional)

### Contributing Code

We welcome code contributions! Here are some areas where you can help:

**High Priority:**
- Core framework implementation
- Codec architecture implementations
- Training pipeline optimization
- Evaluation metrics
- Documentation improvements

**Medium Priority:**
- Cloud provider abstractions (GCP, Azure support)
- Alternative codec architectures
- Performance optimizations
- Unit and integration tests
- Example notebooks and tutorials

**Nice to Have:**
- Web UI for monitoring
- Alternative optimization techniques
- Multi-language support
- Mobile deployment tools

## 🔄 Development Workflow

### Branching Strategy

- `main` - Production-ready code (protected)
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes
- `docs/*` - Documentation updates

### Creating a Feature Branch

```bash
# Update your local repository
git checkout develop
git pull upstream develop

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Run tests
pytest tests/

# Run linters
black .
flake8 .
mypy .

# Commit your changes
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

## 🎨 Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length:** 100 characters (not 79)
- **Formatter:** Black (automatically enforced)
- **Linter:** Flake8
- **Type hints:** Use mypy for type checking

```python
# Good example
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
) -> dict[str, float]:
    """
    Train a neural network model.
    
    Args:
        model: PyTorch model to train
        data_loader: DataLoader with training data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        
    Returns:
        Dictionary with training metrics
    """
    metrics = {}
    
    for epoch in range(epochs):
        loss = train_epoch(model, data_loader, optimizer)
        metrics[f"epoch_{epoch}"] = loss
        
    return metrics
```

### Documentation Style

- **Docstrings:** Google style
- **Comments:** Clear and concise, explain *why* not *what*
- **README updates:** Update relevant docs when adding features

```python
def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image tensor (B, C, H, W)
        reconstructed: Reconstructed image tensor (B, C, H, W)
        
    Returns:
        PSNR value in decibels (dB)
        
    Raises:
        ValueError: If tensor shapes don't match
        
    Example:
        >>> original = torch.randn(1, 3, 256, 256)
        >>> reconstructed = original + 0.1 * torch.randn_like(original)
        >>> psnr = compute_psnr(original, reconstructed)
        >>> print(f"PSNR: {psnr:.2f} dB")
    """
    # Implementation
```

### Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

**Examples:**

```
feat(codec): add VQ-VAE architecture implementation

Implemented vector quantized variational autoencoder for video compression.
Includes codebook learning and commitment loss.

Closes #42
```

```
fix(training): resolve memory leak in data loader

Fixed issue where video frames were not being properly released,
causing OOM errors during long training runs.

Fixes #156
```

```
docs(readme): update installation instructions

Added troubleshooting section and clarified CUDA requirements.
```

## 🔍 Pull Request Process

### Before Submitting

1. **Update your branch** with latest upstream changes
2. **Run all tests** and ensure they pass
3. **Run linters** (black, flake8, mypy)
4. **Update documentation** if needed
5. **Add tests** for new functionality
6. **Update CHANGELOG.md** if applicable

### Submitting a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**:
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template

3. **PR Description Should Include**:
   - Clear title following conventional commits
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots/demos if applicable

4. **PR Review Process**:
   - Automated checks must pass (CI/CD)
   - At least one maintainer approval required
   - Address review comments
   - Maintain up-to-date with develop branch

### PR Template

```markdown
## Description
Brief description of changes

## Related Issue
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Has This Been Tested?
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing unit tests pass locally
- [ ] Any dependent changes have been merged
```

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
import torch
from codec.models import SimpleAutoencoder

def test_autoencoder_forward():
    """Test autoencoder forward pass."""
    model = SimpleAutoencoder(channels=64)
    x = torch.randn(2, 3, 8, 64, 64)  # Batch of video clips
    
    output = model(x)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    

def test_autoencoder_compression():
    """Test that autoencoder compresses representation."""
    model = SimpleAutoencoder(channels=64)
    x = torch.randn(1, 3, 8, 64, 64)
    
    latent = model.encode(x)
    reconstructed = model.decode(latent)
    
    # Latent should be smaller than input
    assert latent.numel() < x.numel()
    # Reconstruction should match input shape
    assert reconstructed.shape == x.shape
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_codec.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

## 📝 Documentation

### Updating Documentation

When adding features, update:
- **README.md** - If changing setup, features, or usage
- **Code docstrings** - All public functions and classes
- **IMPLEMENTATION_PLAN.md** - If changing architecture
- **CODEC_ARCHITECTURE_GUIDE.md** - If adding new codec techniques

### Building Documentation Locally

```bash
cd docs
make html
open _build/html/index.html
```

## 🏆 Recognition

Contributors will be recognized in:
- README.md Contributors section
- CHANGELOG.md for their contributions
- GitHub contributors page

## 💬 Questions?

- **General questions:** Open a GitHub Discussion
- **Bug reports:** Open a GitHub Issue
- **Security issues:** Email directly (see SECURITY.md)

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [CompressAI Documentation](https://interdigitalinc.github.io/CompressAI/)
- [Video Compression Basics](https://en.wikipedia.org/wiki/Video_compression_picture_types)
- [Neural Compression Survey](https://arxiv.org/abs/2202.06533)

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Coding! 🚀**


