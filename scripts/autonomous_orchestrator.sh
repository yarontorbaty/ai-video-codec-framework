#!/bin/bash
# Autonomous Orchestrator Script
# Runs AI codec experiments periodically and logs results to DynamoDB

LOG_FILE="/var/log/ai-codec-orchestrator.log"
EXPERIMENT_SCRIPT="/opt/scripts/real_experiment.py"
INTERVAL_HOURS=6

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== AI Video Codec Autonomous Orchestrator Started ==="
log "Running experiments every $INTERVAL_HOURS hours"

# Run initial experiment immediately
log "Running initial experiment..."
cd /opt
python3 "$EXPERIMENT_SCRIPT" >> "$LOG_FILE" 2>&1
log "Initial experiment completed with exit code: $?"

# Main loop - run experiments periodically
while true; do
    SLEEP_SECONDS=$((INTERVAL_HOURS * 3600))
    log "Sleeping for $INTERVAL_HOURS hours..."
    sleep "$SLEEP_SECONDS"
    
    log "Starting new experiment cycle..."
    python3 "$EXPERIMENT_SCRIPT" >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    log "Experiment completed with exit code: $EXIT_CODE"
    
    # Check for errors and log metrics
    if [ $EXIT_CODE -eq 0 ]; then
        log "✅ Experiment successful"
    else
        log "❌ Experiment failed with code $EXIT_CODE"
    fi
    
    # Log current system metrics
    MEMORY=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2 }')
    CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
    DISK=$(df -h /opt | awk 'NR==2{print $5}')
    log "System metrics - CPU: $CPU, Memory: $MEMORY, Disk: $DISK"
done

