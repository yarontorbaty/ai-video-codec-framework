#!/bin/bash
#
# SSH into AI Video Codec instances
# Supports both SSH (with key) and AWS Systems Manager Session Manager (keyless)
#

set -e

REGION="us-east-1"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 AI Video Codec - Instance SSH Helper${NC}\n"

# Function to check if Session Manager plugin is installed
check_ssm_plugin() {
    if command -v session-manager-plugin &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to list all codec instances
list_instances() {
    echo -e "${YELLOW}📋 Finding codec instances...${NC}\n"
    
    INSTANCES=$(aws ec2 describe-instances \
        --region $REGION \
        --filters "Name=tag:Name,Values=*codec*" "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].[InstanceId,Tags[?Key==`Name`].Value|[0],KeyName,PublicIpAddress,PrivateIpAddress,State.Name]' \
        --output json)
    
    echo "$INSTANCES" | jq -r '.[][] | @tsv' | nl -w2 -s'. '
}

# Function to connect via SSM
connect_ssm() {
    local instance_id=$1
    local instance_name=$2
    
    echo -e "\n${GREEN}🔌 Connecting to ${instance_name} via AWS Systems Manager...${NC}"
    echo -e "${YELLOW}Note: This method doesn't require SSH keys!${NC}\n"
    
    aws ssm start-session \
        --region $REGION \
        --target "$instance_id"
}

# Function to connect via SSH
connect_ssh() {
    local instance_id=$1
    local instance_name=$2
    local public_ip=$3
    local key_name=$4
    
    if [ -z "$public_ip" ] || [ "$public_ip" == "None" ]; then
        echo -e "${RED}❌ No public IP address. Instance may be in private subnet.${NC}"
        echo -e "${YELLOW}💡 Try using SSM Session Manager instead (option 's')${NC}"
        return 1
    fi
    
    # Check for SSH key
    local key_path=""
    if [ -n "$key_name" ] && [ "$key_name" != "None" ]; then
        # Try common locations
        if [ -f ~/.ssh/${key_name}.pem ]; then
            key_path=~/.ssh/${key_name}.pem
        elif [ -f ~/.ssh/${key_name} ]; then
            key_path=~/.ssh/${key_name}
        elif [ -f $(pwd)/${key_name}.pem ]; then
            key_path=$(pwd)/${key_name}.pem
        fi
        
        if [ -z "$key_path" ]; then
            echo -e "${RED}❌ SSH key '${key_name}' not found in:${NC}"
            echo "   - ~/.ssh/${key_name}.pem"
            echo "   - ~/.ssh/${key_name}"
            echo "   - $(pwd)/${key_name}.pem"
            echo ""
            echo -e "${YELLOW}💡 Please specify the key path or use SSM Session Manager (option 's')${NC}"
            read -p "Enter full path to SSH key (or press Enter to skip): " custom_key_path
            if [ -n "$custom_key_path" ] && [ -f "$custom_key_path" ]; then
                key_path=$custom_key_path
            else
                return 1
            fi
        fi
        
        # Ensure correct permissions
        chmod 400 "$key_path" 2>/dev/null
        
        echo -e "\n${GREEN}🔐 Connecting to ${instance_name} via SSH...${NC}"
        echo -e "${BLUE}Using key: ${key_path}${NC}\n"
        
        ssh -i "$key_path" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            ec2-user@${public_ip}
    else
        echo -e "${RED}❌ No SSH key configured for this instance${NC}"
        echo -e "${YELLOW}💡 Use SSM Session Manager instead (option 's')${NC}"
        return 1
    fi
}

# Function to check logs on instance
check_logs() {
    local instance_id=$1
    local log_path=$2
    
    echo -e "\n${GREEN}📜 Fetching logs from instance...${NC}\n"
    
    # Use SSM to run remote command
    aws ssm send-command \
        --region $REGION \
        --instance-ids "$instance_id" \
        --document-name "AWS-RunShellScript" \
        --parameters "commands=[\"tail -100 $log_path\"]" \
        --query 'Command.CommandId' \
        --output text
}

# Main menu
main() {
    # List instances
    echo "$INSTANCES" | jq -r '.[][] | @tsv' | while IFS=$'\t' read -r instance_id name key_name public_ip private_ip state; do
        echo -e "${GREEN}Instance:${NC} $name"
        echo -e "  ${BLUE}ID:${NC} $instance_id"
        echo -e "  ${BLUE}Key:${NC} ${key_name:-None}"
        echo -e "  ${BLUE}Public IP:${NC} ${public_ip:-None}"
        echo -e "  ${BLUE}Private IP:${NC} ${private_ip}"
        echo -e "  ${BLUE}State:${NC} $state"
        echo ""
    done
    
    echo -e "${YELLOW}Available connection methods:${NC}"
    if check_ssm_plugin; then
        echo -e "  ${GREEN}✅ SSM Session Manager (keyless)${NC}"
    else
        echo -e "  ${RED}❌ SSM Session Manager (not installed)${NC}"
        echo -e "     Install with: ${BLUE}brew install --cask session-manager-plugin${NC}"
    fi
    echo -e "  ${GREEN}✅ SSH (requires key)${NC}"
    echo ""
    
    # Get user choice
    echo -e "${BLUE}Select instance to connect to:${NC}"
    echo "$INSTANCES" | jq -r '.[][] | @tsv' | nl -w2 -s'. ' | while read line; do
        echo "$line"
    done
    echo ""
    
    read -p "Enter instance number: " choice
    
    # Parse selected instance
    SELECTED=$(echo "$INSTANCES" | jq -r ".[][$((choice-1))] | @tsv")
    
    if [ -z "$SELECTED" ]; then
        echo -e "${RED}❌ Invalid selection${NC}"
        exit 1
    fi
    
    read -r INSTANCE_ID NAME KEY_NAME PUBLIC_IP PRIVATE_IP STATE <<< "$SELECTED"
    
    echo ""
    echo -e "${BLUE}Connection method:${NC}"
    echo "  1. SSH (requires key '$KEY_NAME')"
    if check_ssm_plugin; then
        echo "  2. SSM Session Manager (keyless)"
    else
        echo "  2. SSM Session Manager (not installed - use 'brew install --cask session-manager-plugin')"
    fi
    echo "  3. Check recent logs"
    echo "  4. Cancel"
    echo ""
    
    read -p "Choose method [1-4]: " method
    
    case $method in
        1)
            connect_ssh "$INSTANCE_ID" "$NAME" "$PUBLIC_IP" "$KEY_NAME"
            ;;
        2)
            if check_ssm_plugin; then
                connect_ssm "$INSTANCE_ID" "$NAME"
            else
                echo -e "${RED}❌ SSM Session Manager plugin not installed${NC}"
                echo -e "${YELLOW}Install with: brew install --cask session-manager-plugin${NC}"
                exit 1
            fi
            ;;
        3)
            echo "Select log to view:"
            echo "  1. /tmp/codec_versions/ (LLM code tests)"
            echo "  2. /var/log/cloud-init-output.log (System logs)"
            echo "  3. Orchestrator service logs"
            read -p "Choice: " log_choice
            case $log_choice in
                1) check_logs "$INSTANCE_ID" "/tmp/codec_versions/" ;;
                2) check_logs "$INSTANCE_ID" "/var/log/cloud-init-output.log" ;;
                3) 
                    echo "Fetching service logs..."
                    aws ssm send-command \
                        --region $REGION \
                        --instance-ids "$INSTANCE_ID" \
                        --document-name "AWS-RunShellScript" \
                        --parameters 'commands=["sudo journalctl -u ai-video-codec-orchestrator -n 100"]'
                    ;;
            esac
            ;;
        4)
            echo "Cancelled"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid choice${NC}"
            exit 1
            ;;
    esac
}

# Quick access functions
case "${1:-}" in
    --orchestrator|-o)
        ORCH_ID=$(aws ec2 describe-instances \
            --region $REGION \
            --filters "Name=tag:Name,Values=*orchestrator*" "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].InstanceId' \
            --output text)
        
        ORCH_IP=$(aws ec2 describe-instances \
            --region $REGION \
            --instance-ids $ORCH_ID \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text)
        
        echo -e "${GREEN}🚀 Quick connect to orchestrator...${NC}\n"
        
        if check_ssm_plugin; then
            connect_ssm "$ORCH_ID" "orchestrator"
        else
            connect_ssh "$ORCH_ID" "orchestrator" "$ORCH_IP" "bobov"
        fi
        ;;
    --experiment|-e)
        EXP_ID=$(aws ec2 describe-instances \
            --region $REGION \
            --filters "Name=tag:Name,Values=*experiment*" "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].InstanceId' \
            --output text)
        
        EXP_IP=$(aws ec2 describe-instances \
            --region $REGION \
            --instance-ids $EXP_ID \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text)
        
        echo -e "${GREEN}🧪 Quick connect to experiment instance...${NC}\n"
        
        if check_ssm_plugin; then
            connect_ssm "$EXP_ID" "experiment"
        else
            connect_ssh "$EXP_ID" "experiment" "$EXP_IP" "bobov"
        fi
        ;;
    --logs|-l)
        echo -e "${YELLOW}📜 Checking LLM code test logs...${NC}\n"
        ORCH_ID=$(aws ec2 describe-instances \
            --region $REGION \
            --filters "Name=tag:Name,Values=*orchestrator*" "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].InstanceId' \
            --output text)
        
        echo "Running command to list recent code test failures..."
        CMD_ID=$(aws ssm send-command \
            --region $REGION \
            --instance-ids "$ORCH_ID" \
            --document-name "AWS-RunShellScript" \
            --parameters 'commands=["ls -lt /tmp/codec_versions/validation_failure_*.txt 2>/dev/null | head -5 || echo \"No failures found\"","echo \"---\"","[ -f /tmp/codec_versions/validation_failure_*.txt ] && cat $(ls -t /tmp/codec_versions/validation_failure_*.txt | head -1) || echo \"No failure logs\""]' \
            --query 'Command.CommandId' \
            --output text)
        
        echo "Command ID: $CMD_ID"
        echo "Waiting for results..."
        sleep 3
        
        aws ssm get-command-invocation \
            --region $REGION \
            --command-id "$CMD_ID" \
            --instance-id "$ORCH_ID" \
            --query 'StandardOutputContent' \
            --output text
        ;;
    --help|-h|*)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Interactive mode (no options):"
        echo "  Shows all instances and lets you choose"
        echo ""
        echo "Quick access options:"
        echo "  -o, --orchestrator    Connect to orchestrator instance"
        echo "  -e, --experiment      Connect to experiment instance"
        echo "  -l, --logs            Check LLM code test logs"
        echo "  -h, --help            Show this help"
        echo ""
        echo "Examples:"
        echo "  $0                    # Interactive mode"
        echo "  $0 --orchestrator     # Quick connect to orchestrator"
        echo "  $0 --logs             # View LLM test failure logs"
        echo ""
        
        if [ "${1:-}" != "--help" ] && [ "${1:-}" != "-h" ]; then
            main
        fi
        ;;
esac

