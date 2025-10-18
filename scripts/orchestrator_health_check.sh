#!/bin/bash
# Neural Codec Orchestrator Health Check
set -e

ORCHESTRATOR_URL="http://34.239.1.29:8081"
LOG_FILE="/tmp/orchestrator_health.log"

echo "$(date): Starting orchestrator health check" >> $LOG_FILE

# Check if orchestrator is responding
if curl -s -f "$ORCHESTRATOR_URL/health" > /dev/null; then
    echo "$(date): Orchestrator health check PASSED" >> $LOG_FILE
    
    # Get orchestrator status
    HEALTH_DATA=$(curl -s "$ORCHESTRATOR_URL/health")
    AVAILABLE_WORKERS=$(echo "$HEALTH_DATA" | python3 -c "import sys,json; print(len([w for w in json.load(sys.stdin)['available_workers'] if w['status']['status'] == 'healthy']))")
    
    if [ "$AVAILABLE_WORKERS" -gt 0 ]; then
        echo "$(date): Orchestrator has $AVAILABLE_WORKERS available workers" >> $LOG_FILE
        exit 0
    else
        echo "$(date): Orchestrator has no available workers" >> $LOG_FILE
        exit 1
    fi
else
    echo "$(date): Orchestrator health check FAILED - not responding" >> $LOG_FILE
    exit 1
fi
