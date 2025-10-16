#!/bin/bash
# Real-Time Autonomous Orchestrator with LLM Planning
# Continuously monitors health and reacts to experiment results in real-time

LOG_FILE="/var/log/ai-codec-orchestrator.log"
EXPERIMENT_SCRIPT="/opt/scripts/real_experiment.py"
PLANNER_SCRIPT="/opt/src/agents/llm_experiment_planner.py"
DELAY_BETWEEN_EXPERIMENTS=60  # 1 minute between experiments for real-time monitoring
MAX_EXPERIMENT_DURATION=600    # 10 minutes timeout per experiment

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Real-Time AI Video Codec Orchestrator Started ==="
log "Continuously running experiments with real-time LLM planning"
log "Delay between experiments: ${DELAY_BETWEEN_EXPERIMENTS}s"

# Fetch API key from AWS Secrets Manager
if [ -z "$ANTHROPIC_API_KEY" ]; then
    log "Fetching API key from AWS Secrets Manager..."
    SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id ai-video-codec/anthropic-api-key --region us-east-1 --query SecretString --output text 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$SECRET_JSON" ]; then
        export ANTHROPIC_API_KEY=$(echo "$SECRET_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['ANTHROPIC_API_KEY'])")
        log "✅ API key retrieved from Secrets Manager"
    else
        log "⚠️  WARNING: Could not retrieve API key from Secrets Manager"
    fi
fi

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    log "⚠️  WARNING: ANTHROPIC_API_KEY not set - LLM analysis will show 'not available'"
else
    log "✅ LLM API key detected - Claude-powered real-time planning enabled"
fi

# Monitor system health
check_health() {
    MEMORY=$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }')
    CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    DISK=$(df -h /opt | awk 'NR==2{print $5}' | tr -d '%')
    
    # Alert if resources are constrained
    if (( $(echo "$MEMORY > 90" | bc -l) )); then
        log "⚠️  HIGH MEMORY USAGE: ${MEMORY}%"
    fi
    if (( $(echo "$CPU > 90" | bc -l) )); then
        log "⚠️  HIGH CPU USAGE: ${CPU}%"
    fi
    if (( DISK > 90 )); then
        log "⚠️  HIGH DISK USAGE: ${DISK}%"
    fi
    
    log "Health: CPU=${CPU}%, Memory=${MEMORY}%, Disk=${DISK}%"
}

# Run experiment with timeout and health monitoring
run_experiment_with_monitoring() {
    local ITERATION=$1
    
    log "=========================================="
    log "ITERATION $ITERATION - STARTING"
    log "=========================================="
    
    # Pre-experiment health check
    check_health
    
    # Run experiment with timeout
    log "🔬 Running experiment $ITERATION..."
    cd /opt
    timeout $MAX_EXPERIMENT_DURATION python3 "$EXPERIMENT_SCRIPT" >> "$LOG_FILE" 2>&1 &
    EXPERIMENT_PID=$!
    
    # Monitor experiment in real-time
    ELAPSED=0
    while kill -0 $EXPERIMENT_PID 2>/dev/null; do
        sleep 10
        ELAPSED=$((ELAPSED + 10))
        if [ $((ELAPSED % 60)) -eq 0 ]; then
            log "Experiment $ITERATION running for ${ELAPSED}s..."
        fi
    done
    
    wait $EXPERIMENT_PID
    EXPERIMENT_EXIT_CODE=$?
    
    if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
        log "✅ Experiment $ITERATION completed successfully"
    elif [ $EXPERIMENT_EXIT_CODE -eq 124 ]; then
        log "⏱️  Experiment $ITERATION timed out after ${MAX_EXPERIMENT_DURATION}s"
    else
        log "❌ Experiment $ITERATION failed with exit code $EXPERIMENT_EXIT_CODE"
    fi
    
    # Post-experiment health check
    check_health
    
    log "=========================================="
    log "ITERATION $ITERATION - COMPLETED"
    log "=========================================="
    
    return $EXPERIMENT_EXIT_CODE
}

# Main continuous loop
ITERATION=1
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=5

while true; do
    # Run experiment with real-time monitoring
    run_experiment_with_monitoring $ITERATION
    RESULT=$?
    
    # Track failures
    if [ $RESULT -ne 0 ]; then
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        log "⚠️  Consecutive failures: $CONSECUTIVE_FAILURES"
        
        if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
            log "🚨 CRITICAL: $MAX_CONSECUTIVE_FAILURES consecutive failures. Entering recovery mode..."
            log "Waiting 5 minutes before resuming..."
            sleep 300
            CONSECUTIVE_FAILURES=0
        fi
    else
        CONSECUTIVE_FAILURES=0
    fi
    
    # Short delay before next experiment for real-time operation
    log "Waiting ${DELAY_BETWEEN_EXPERIMENTS}s before next experiment..."
    sleep $DELAY_BETWEEN_EXPERIMENTS
    
    ITERATION=$((ITERATION + 1))
done

