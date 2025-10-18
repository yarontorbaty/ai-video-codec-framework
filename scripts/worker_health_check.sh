#!/bin/bash
# Neural Codec Worker Health Check
set -e

WORKER_URL="http://18.208.180.67:8080"
LOG_FILE="/tmp/worker_health.log"

echo "$(date): Starting worker health check" >> $LOG_FILE

# Check if worker is responding
if curl -s -f "$WORKER_URL/health" > /dev/null; then
    echo "$(date): Worker health check PASSED" >> $LOG_FILE
    
    # Get worker status
    STATUS=$(curl -s "$WORKER_URL/status" | python3 -c "import sys,json; print(json.load(sys.stdin)['is_processing'])")
    
    if [ "$STATUS" = "false" ]; then
        echo "$(date): Worker is idle - ready for experiments" >> $LOG_FILE
        exit 0
    else
        echo "$(date): Worker is busy processing experiment" >> $LOG_FILE
        exit 0
    fi
else
    echo "$(date): Worker health check FAILED - not responding" >> $LOG_FILE
    exit 1
fi
