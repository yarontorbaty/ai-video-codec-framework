#!/bin/bash
# Procedural Autonomous Orchestrator with LLM Planning and Self-Healing
# No time windows - experiments run through complete validation/execution/fix cycles

BASE_DIR="/home/ec2-user/ai-video-codec"
LOG_FILE="/tmp/orch.log"
EXPERIMENT_SCRIPT="$BASE_DIR/src/agents/procedural_experiment_runner.py"
PLANNER_SCRIPT="$BASE_DIR/src/agents/llm_experiment_planner.py"
# No delays - each experiment runs to completion (validation + fixes + execution)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Procedural AI Video Codec Orchestrator Started ==="
log "Running experiments through complete procedural cycles:"
log "  1. Design experiment and code"
log "  2. Deploy to sandbox"
log "  3. Validate (retry with fixes)"
log "  4. Execute (retry with fixes)"
log "  5. Analyze results"
log "  6. Design next experiment"

# Fetch API key from AWS Secrets Manager
if [ -z "$ANTHROPIC_API_KEY" ]; then
    log "Fetching API key from AWS Secrets Manager..."
    SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id ai-video-codec/anthropic-api-key --region us-east-1 --query SecretString --output text 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$SECRET_JSON" ]; then
        export ANTHROPIC_API_KEY=$(echo "$SECRET_JSON" | python3 -c "import sys, json; data = json.loads(sys.stdin.read()); print(data['ANTHROPIC_API_KEY'])" 2>/dev/null)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            # Fallback: try treating it as plain text
            export ANTHROPIC_API_KEY="$SECRET_JSON"
        fi
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

# Run experiment with procedural validation/execution/fix cycle
run_procedural_experiment() {
    local ITERATION=$1
    
    log "=========================================="
    log "ITERATION $ITERATION - STARTING"
    log "=========================================="
    
    # Pre-experiment health check
    check_health
    
    # Run experiment - no timeout, runs until complete or max retries
    log "🔬 Running procedural experiment $ITERATION..."
    log "  This may take a while as it validates, executes, and fixes issues..."
    
    cd "$BASE_DIR"
    export EXPERIMENT_ITERATION=$ITERATION
    
    # Run without timeout - procedural runner handles its own retry logic
    python3 "$EXPERIMENT_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    EXPERIMENT_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
        log "✅ Experiment $ITERATION completed successfully"
    else
        log "❌ Experiment $ITERATION failed with exit code $EXPERIMENT_EXIT_CODE"
        log "  Check logs above for human intervention requirements"
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
    # Run procedural experiment (no timeout, handles own retry logic)
    run_procedural_experiment $ITERATION
    RESULT=$?
    
    # Track failures
    if [ $RESULT -ne 0 ]; then
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        log "⚠️  Consecutive failures: $CONSECUTIVE_FAILURES"
        
        if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
            log "🚨 CRITICAL: $MAX_CONSECUTIVE_FAILURES consecutive failures detected"
            log "🚨 HUMAN INTERVENTION MAY BE REQUIRED"
            log "Entering recovery mode - waiting 5 minutes..."
            sleep 300
            CONSECUTIVE_FAILURES=0
        else
            # Brief pause between failed attempts
            log "Pausing 30s before retry..."
            sleep 30
        fi
    else
        CONSECUTIVE_FAILURES=0
        # Brief pause between successful experiments (for monitoring)
        log "Pausing 10s before next experiment..."
        sleep 10
    fi
    
    ITERATION=$((ITERATION + 1))
done

