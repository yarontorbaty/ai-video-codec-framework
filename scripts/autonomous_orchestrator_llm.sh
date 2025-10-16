#!/bin/bash
# Autonomous Orchestrator with LLM Planning
# Uses LLM to analyze results and plan improved experiments

LOG_FILE="/var/log/ai-codec-orchestrator.log"
EXPERIMENT_SCRIPT="/opt/scripts/real_experiment.py"
PLANNER_SCRIPT="/opt/src/agents/llm_experiment_planner.py"
INTERVAL_HOURS=6

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== AI Video Codec LLM-Powered Autonomous Orchestrator Started ==="
log "Running experiments every $INTERVAL_HOURS hours with LLM planning"

# Check if ANTHROPIC_API_KEY is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    log "⚠️  WARNING: ANTHROPIC_API_KEY not set - will use fallback rule-based planning"
else
    log "✅ LLM API key detected - Claude-powered planning enabled"
fi

# Run initial experiment immediately
log "Running initial experiment..."
cd /opt
python3 "$EXPERIMENT_SCRIPT" >> "$LOG_FILE" 2>&1
EXPERIMENT_EXIT_CODE=$?
log "Initial experiment completed with exit code: $EXPERIMENT_EXIT_CODE"

# Main loop - analyze and run improved experiments
ITERATION=1
while true; do
    SLEEP_SECONDS=$((INTERVAL_HOURS * 3600))
    log "Sleeping for $INTERVAL_HOURS hours before next iteration..."
    sleep "$SLEEP_SECONDS"
    
    log "=========================================="
    log "ITERATION $ITERATION - LLM PLANNING PHASE"
    log "=========================================="
    
    # Run LLM analysis and planning
    log "🤖 Analyzing previous experiments with LLM..."
    python3 "$PLANNER_SCRIPT" >> "$LOG_FILE" 2>&1
    PLANNER_EXIT_CODE=$?
    
    if [ $PLANNER_EXIT_CODE -eq 0 ]; then
        log "✅ LLM planning completed successfully"
    else
        log "⚠️  LLM planning failed (code: $PLANNER_EXIT_CODE) - continuing with standard experiment"
    fi
    
    log "=========================================="
    log "ITERATION $ITERATION - EXPERIMENT PHASE"
    log "=========================================="
    
    # Run experiment (potentially with modified parameters based on LLM suggestions)
    log "🔬 Starting experiment cycle $ITERATION..."
    python3 "$EXPERIMENT_SCRIPT" >> "$LOG_FILE" 2>&1
    EXPERIMENT_EXIT_CODE=$?
    
    if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
        log "✅ Experiment $ITERATION successful"
    else
        log "❌ Experiment $ITERATION failed with code $EXPERIMENT_EXIT_CODE"
    fi
    
    # Log system metrics
    MEMORY=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2 }')
    CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
    DISK=$(df -h /opt | awk 'NR==2{print $5}')
    log "System metrics - CPU: $CPU, Memory: $MEMORY, Disk: $DISK"
    
    ITERATION=$((ITERATION + 1))
    log "Completed iteration $ITERATION. Total iterations: $ITERATION"
done

