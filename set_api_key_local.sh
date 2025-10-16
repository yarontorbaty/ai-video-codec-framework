#!/bin/bash
# Set Anthropic API Key on EC2 Instance

echo "Paste your Anthropic API key:"
read -s API_KEY

if [ -z "$API_KEY" ]; then
    echo "Error: No API key provided"
    exit 1
fi

echo ""
echo "Setting API key on EC2 instance..."

# Clean old keys and set new one with proper newline
CMD="sudo sed -i /ANTHROPIC_API_KEY/d /root/.bashrc && echo export ANTHROPIC_API_KEY=$API_KEY >> /root/.bashrc && echo API key set"

COMMAND_ID=$(aws ssm send-command \
  --instance-ids i-063947ae46af6dbf8 \
  --document-name "AWS-RunShellScript" \
  --parameters commands="$CMD" \
  --output text \
  --query 'Command.CommandId')

echo "Command ID: $COMMAND_ID"
sleep 5

aws ssm get-command-invocation \
  --command-id $COMMAND_ID \
  --instance-id i-063947ae46af6dbf8 \
  --query 'StandardOutputContent' \
  --output text

unset API_KEY
echo ""
echo "Done!"
