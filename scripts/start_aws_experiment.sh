#!/bin/bash
set -e

# AI Video Codec Framework - Start AWS Experiment
# This script starts the AI codec experiment on AWS EC2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Video Codec - Start AWS Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
PROJECT_NAME="ai-video-codec"
ENVIRONMENT="production"
REGION="us-east-1"
INSTANCE_ID="i-0cef8f1c45569dfe9"

echo -e "${BLUE}Instance ID: ${INSTANCE_ID}${NC}"
echo -e "${BLUE}Region: ${REGION}${NC}"
echo ""

# Check if instance is running
echo -e "${BLUE}Checking instance status...${NC}"
INSTANCE_STATE=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text)
echo -e "${GREEN}Instance state: ${INSTANCE_STATE}${NC}"

if [ "$INSTANCE_STATE" != "running" ]; then
    echo -e "${RED}Instance is not running. Current state: ${INSTANCE_STATE}${NC}"
    exit 1
fi

# Get instance details
echo -e "${BLUE}Getting instance details...${NC}"
INSTANCE_INFO=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0]' --output json)
echo -e "${GREEN}Instance details retrieved${NC}"

# Check if we can connect via SSM (Systems Manager)
echo -e "${BLUE}Checking SSM connectivity...${NC}"
if aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" --query 'InstanceInformationList[0].InstanceId' --output text 2>/dev/null | grep -q "$INSTANCE_ID"; then
    echo -e "${GREEN}✓ SSM connectivity available${NC}"
    
    # Run command via SSM
    echo -e "${BLUE}Starting AI experiment via SSM...${NC}"
    COMMAND_ID=$(aws ssm send-command \
        --instance-ids $INSTANCE_ID \
        --document-name "AWS-RunShellScript" \
        --parameters 'commands=["cd /opt/ai-video-codec && python3 -m src.agents.experiment_orchestrator --action start"]' \
        --query 'Command.CommandId' \
        --output text)
    
    echo -e "${GREEN}Command sent. Command ID: ${COMMAND_ID}${NC}"
    
    # Monitor command execution
    echo -e "${BLUE}Monitoring command execution...${NC}"
    for i in {1..30}; do
        STATUS=$(aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID --query 'Status' --output text 2>/dev/null || echo "InProgress")
        echo -e "${YELLOW}Status: ${STATUS}${NC}"
        
        if [ "$STATUS" = "Success" ]; then
            echo -e "${GREEN}✓ Command completed successfully${NC}"
            break
        elif [ "$STATUS" = "Failed" ]; then
            echo -e "${RED}✗ Command failed${NC}"
            aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID --query 'StandardErrorContent' --output text
            exit 1
        fi
        
        sleep 10
    done
    
    # Get command output
    echo -e "${BLUE}Getting command output...${NC}"
    aws ssm get-command-invocation --command-id $COMMAND_ID --instance-id $INSTANCE_ID --query 'StandardOutputContent' --output text
    
else
    echo -e "${YELLOW}SSM not available. Instance may still be starting up.${NC}"
    echo -e "${BLUE}You can check the experiment status later with:${NC}"
    echo "aws s3 ls s3://ai-video-codec-videos-580473065386/results/"
    echo "aws dynamodb scan --table-name ai-video-codec-experiments"
fi

# Check for results
echo -e "${BLUE}Checking for experiment results...${NC}"
sleep 5

# Check S3 for results
RESULTS=$(aws s3 ls s3://ai-video-codec-videos-580473065386/results/ --recursive 2>/dev/null | wc -l)
if [ "$RESULTS" -gt 0 ]; then
    echo -e "${GREEN}✓ Found ${RESULTS} result files in S3${NC}"
    aws s3 ls s3://ai-video-codec-videos-580473065386/results/ --recursive
else
    echo -e "${YELLOW}No results found yet. Experiment may still be running.${NC}"
fi

# Check DynamoDB for experiment records
EXPERIMENTS=$(aws dynamodb scan --table-name ai-video-codec-experiments --query 'Count' --output text 2>/dev/null || echo "0")
if [ "$EXPERIMENTS" -gt 0 ]; then
    echo -e "${GREEN}✓ Found ${EXPERIMENTS} experiment records in DynamoDB${NC}"
    aws dynamodb scan --table-name ai-video-codec-experiments --query 'Items[0]' --output json
else
    echo -e "${YELLOW}No experiment records found yet.${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}AWS Experiment Status Check Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Monitor the dashboard for real-time metrics"
echo "2. Check S3 for result files: aws s3 ls s3://ai-video-codec-videos-580473065386/results/"
echo "3. Check DynamoDB for experiment status: aws dynamodb scan --table-name ai-video-codec-experiments"
echo "4. The experiment will run for 1-2 hours on the GPU instance"
echo ""
echo -e "${GREEN}AI Codec Experiment is running on AWS! 🎬🚀${NC}"
