#!/bin/bash
set -e

# AI Video Codec Framework - Environment Setup Script
# This script sets up the local development environment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python 3.10+ is installed
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.10+ required, found ${PYTHON_VERSION}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    echo "Please install pip3"
    exit 1
fi

echo -e "${GREEN}✓ pip3 found${NC}"

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install core dependencies
echo -e "${BLUE}Installing core dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Install development dependencies (optional)
read -p "Install development dependencies? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Installing development dependencies...${NC}"
    pip install -r requirements-dev.txt
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Install pre-commit hooks
if [ -f "requirements-dev.txt" ]; then
    echo -e "${BLUE}Installing pre-commit hooks...${NC}"
    pre-commit install
    echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
fi

# Create config directory if it doesn't exist
if [ ! -d "config" ]; then
    mkdir -p config
    echo -e "${GREEN}✓ Config directory created${NC}"
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    mkdir -p data
    echo -e "${GREEN}✓ Data directory created${NC}"
fi

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo -e "${GREEN}✓ Logs directory created${NC}"
fi

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    mkdir -p models
    echo -e "${GREEN}✓ Models directory created${NC}"
fi

# Check if AWS CLI is installed
echo -e "${BLUE}Checking AWS CLI...${NC}"
if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}Warning: AWS CLI not found${NC}"
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    echo "Then run: aws configure"
else
    echo -e "${GREEN}✓ AWS CLI found${NC}"
    
    # Check if AWS is configured
    if aws sts get-caller-identity > /dev/null 2>&1; then
        echo -e "${GREEN}✓ AWS CLI configured${NC}"
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        echo "Account ID: ${ACCOUNT_ID}"
    else
        echo -e "${YELLOW}Warning: AWS CLI not configured${NC}"
        echo "Please run: aws configure"
    fi
fi

# Check if Docker is installed (for local development)
echo -e "${BLUE}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker not found${NC}"
    echo "Docker is optional for local development"
else
    echo -e "${GREEN}✓ Docker found${NC}"
    
    # Check if Docker is running
    if docker info > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Docker is running${NC}"
    else
        echo -e "${YELLOW}Warning: Docker is not running${NC}"
        echo "Please start Docker Desktop"
    fi
fi

# Check if Git is installed
echo -e "${BLUE}Checking Git...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: Git is not installed${NC}"
    echo "Please install Git"
    exit 1
fi

echo -e "${GREEN}✓ Git found${NC}"

# Check if FFmpeg is installed (for video processing)
echo -e "${BLUE}Checking FFmpeg...${NC}"
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}Warning: FFmpeg not found${NC}"
    echo "FFmpeg is required for video processing"
    echo "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)"
else
    echo -e "${GREEN}✓ FFmpeg found${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cat > .env << EOF
# AI Video Codec Framework Environment Variables
PYTHONPATH=/Users/yarontorbaty/Documents/Code/AiV1
LOG_LEVEL=INFO
ENVIRONMENT=development
AWS_DEFAULT_REGION=us-east-1
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
fi

# Run tests to verify installation
echo -e "${BLUE}Running tests to verify installation...${NC}"
if [ -d "tests" ]; then
    python -m pytest tests/ -v
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${YELLOW}No tests found${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure AWS: aws configure"
echo "3. Copy config template: cp config/aws_config.yaml.template config/aws_config.yaml"
echo "4. Edit config: nano config/aws_config.yaml"
echo "5. Deploy to AWS: ./scripts/deploy_aws.sh"
echo ""
echo -e "${BLUE}Development Commands:${NC}"
echo "- Run tests: pytest"
echo "- Format code: black ."
echo "- Lint code: flake8 ."
echo "- Type check: mypy ."
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
