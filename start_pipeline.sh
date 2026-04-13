#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
nohup python3 main_pipeline.py > "$LOG_FILE" 2>&1 &
PID=$!
echo "Pipeline started in background. PID=$PID"
echo "Log file: $LOG_FILE"
