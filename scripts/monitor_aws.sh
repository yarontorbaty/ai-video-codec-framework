#!/bin/bash
set -e

# AI Video Codec Framework - AWS Monitoring Script
# This script monitors the deployed infrastructure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-video-codec"
ENVIRONMENT="production"
REGION="us-east-1"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec Framework - AWS Monitoring${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}Error: AWS CLI not configured${NC}"
    exit 1
fi

# Get orchestrator IP
ORCHESTRATOR_IP=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME}-compute \
    --query 'Stacks[0].Outputs[?OutputKey==`OrchestratorPublicIP`].OutputValue' \
    --output text \
    --region ${REGION} 2>/dev/null || echo "N/A")

if [ "$ORCHESTRATOR_IP" = "N/A" ] || [ -z "$ORCHESTRATOR_IP" ]; then
    echo -e "${RED}Error: Could not get orchestrator IP. Is the stack deployed?${NC}"
    exit 1
fi

echo -e "${GREEN}Orchestrator IP: ${ORCHESTRATOR_IP}${NC}"
echo ""

# Function to check orchestrator status
check_orchestrator() {
    echo -e "${BLUE}Checking orchestrator status...${NC}"
    
    # Check if orchestrator is running
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ec2-user@${ORCHESTRATOR_IP} "sudo systemctl is-active ai-video-codec-orchestrator" 2>/dev/null | grep -q "active"; then
        echo -e "${GREEN}✓ Orchestrator is running${NC}"
    else
        echo -e "${RED}✗ Orchestrator is not running${NC}"
        return 1
    fi
    
    # Check orchestrator logs
    echo -e "${BLUE}Recent orchestrator logs:${NC}"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ec2-user@${ORCHESTRATOR_IP} "sudo journalctl -u ai-video-codec-orchestrator --no-pager -n 10" 2>/dev/null || echo "Could not fetch logs"
    echo ""
}

# Function to check SQS queues
check_queues() {
    echo -e "${BLUE}Checking SQS queues...${NC}"
    
    # Get queue URLs
    TRAINING_QUEUE_URL=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME}-compute \
        --query 'Stacks[0].Outputs[?OutputKey==`TrainingQueueUrl`].OutputValue' \
        --output text \
        --region ${REGION} 2>/dev/null)
    
    EVALUATION_QUEUE_URL=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME}-compute \
        --query 'Stacks[0].Outputs[?OutputKey==`EvaluationQueueUrl`].OutputValue' \
        --output text \
        --region ${REGION} 2>/dev/null)
    
    if [ "$TRAINING_QUEUE_URL" != "None" ] && [ -n "$TRAINING_QUEUE_URL" ]; then
        TRAINING_MESSAGES=$(aws sqs get-queue-attributes \
            --queue-url ${TRAINING_QUEUE_URL} \
            --attribute-names ApproximateNumberOfMessages \
            --query 'Attributes.ApproximateNumberOfMessages' \
            --output text \
            --region ${REGION} 2>/dev/null || echo "0")
        echo -e "${GREEN}Training Queue: ${TRAINING_MESSAGES} messages${NC}"
    else
        echo -e "${RED}Could not get training queue URL${NC}"
    fi
    
    if [ "$EVALUATION_QUEUE_URL" != "None" ] && [ -n "$EVALUATION_QUEUE_URL" ]; then
        EVALUATION_MESSAGES=$(aws sqs get-queue-attributes \
            --queue-url ${EVALUATION_QUEUE_URL} \
            --attribute-names ApproximateNumberOfMessages \
            --query 'Attributes.ApproximateNumberOfMessages' \
            --output text \
            --region ${REGION} 2>/dev/null || echo "0")
        echo -e "${GREEN}Evaluation Queue: ${EVALUATION_MESSAGES} messages${NC}"
    else
        echo -e "${RED}Could not get evaluation queue URL${NC}"
    fi
    echo ""
}

# Function to check DynamoDB tables
check_dynamodb() {
    echo -e "${BLUE}Checking DynamoDB tables...${NC}"
    
    # Check experiments table
    EXPERIMENTS_COUNT=$(aws dynamodb scan \
        --table-name ${PROJECT_NAME}-experiments \
        --select COUNT \
        --query 'Count' \
        --output text \
        --region ${REGION} 2>/dev/null || echo "0")
    echo -e "${GREEN}Experiments: ${EXPERIMENTS_COUNT} records${NC}"
    
    # Check metrics table
    METRICS_COUNT=$(aws dynamodb scan \
        --table-name ${PROJECT_NAME}-metrics \
        --select COUNT \
        --query 'Count' \
        --output text \
        --region ${REGION} 2>/dev/null || echo "0")
    echo -e "${GREEN}Metrics: ${METRICS_COUNT} records${NC}"
    echo ""
}

# Function to check S3 buckets
check_s3() {
    echo -e "${BLUE}Checking S3 buckets...${NC}"
    
    # Get bucket names
    ARTIFACTS_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME}-storage \
        --query 'Stacks[0].Outputs[?OutputKey==`ArtifactsBucketName`].OutputValue' \
        --output text \
        --region ${REGION} 2>/dev/null)
    
    if [ "$ARTIFACTS_BUCKET" != "None" ] && [ -n "$ARTIFACTS_BUCKET" ]; then
        ARTIFACTS_OBJECTS=$(aws s3 ls s3://${ARTIFACTS_BUCKET}/ --recursive | wc -l)
        echo -e "${GREEN}Artifacts Bucket: ${ARTIFACTS_OBJECTS} objects${NC}"
    else
        echo -e "${RED}Could not get artifacts bucket name${NC}"
    fi
    echo ""
}

# Function to check EC2 instances
check_instances() {
    echo -e "${BLUE}Checking EC2 instances...${NC}"
    
    # Get all instances with the project tag
    INSTANCES=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=${PROJECT_NAME}*" "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,PublicIpAddress,Tags[?Key==`Name`].Value|[0]]' \
        --output table \
        --region ${REGION} 2>/dev/null)
    
    if [ -n "$INSTANCES" ]; then
        echo "$INSTANCES"
    else
        echo -e "${YELLOW}No running instances found${NC}"
    fi
    echo ""
}

# Function to check costs
check_costs() {
    echo -e "${BLUE}Checking costs...${NC}"
    
    # Get current month costs
    CURRENT_MONTH=$(date +%Y-%m-01)
    NEXT_MONTH=$(date -d "$(date +%Y-%m-01) +1 month" +%Y-%m-01)
    
    COST=$(aws ce get-cost-and-usage \
        --time-period Start=${CURRENT_MONTH},End=${NEXT_MONTH} \
        --granularity MONTHLY \
        --metrics UnblendedCost \
        --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
        --output text \
        --region ${REGION} 2>/dev/null || echo "0")
    
    if [ "$COST" != "0" ]; then
        echo -e "${GREEN}Current month cost: \$${COST}${NC}"
    else
        echo -e "${YELLOW}Could not retrieve cost information${NC}"
    fi
    echo ""
}

# Function to show recent experiments
show_experiments() {
    echo -e "${BLUE}Recent experiments...${NC}"
    
    # Get recent experiments from DynamoDB
    EXPERIMENTS=$(aws dynamodb scan \
        --table-name ${PROJECT_NAME}-experiments \
        --limit 5 \
        --query 'Items[*].[experiment_id.S,status.S,created_at.S]' \
        --output table \
        --region ${REGION} 2>/dev/null)
    
    if [ -n "$EXPERIMENTS" ]; then
        echo "$EXPERIMENTS"
    else
        echo -e "${YELLOW}No experiments found${NC}"
    fi
    echo ""
}

# Main monitoring loop
if [ "$1" = "--watch" ]; then
    echo -e "${BLUE}Starting continuous monitoring (Ctrl+C to stop)...${NC}"
    echo ""
    
    while true; do
        clear
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}AI Video Codec Framework - Live Monitor${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo "Last updated: $(date)"
        echo ""
        
        check_orchestrator
        check_queues
        check_dynamodb
        check_s3
        check_instances
        check_costs
        show_experiments
        
        echo -e "${BLUE}Refreshing in 30 seconds...${NC}"
        sleep 30
    done
else
    # Single check
    check_orchestrator
    check_queues
    check_dynamodb
    check_s3
    check_instances
    check_costs
    show_experiments
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Monitoring complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo "- Continuous monitoring: $0 --watch"
    echo "- SSH to orchestrator: ssh ec2-user@${ORCHESTRATOR_IP}"
    echo "- View orchestrator logs: ssh ec2-user@${ORCHESTRATOR_IP} 'sudo journalctl -u ai-video-codec-orchestrator -f'"
    echo "- AWS Console: https://console.aws.amazon.com/"
fi
