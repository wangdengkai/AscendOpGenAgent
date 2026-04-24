#!/bin/bash
# Deploy and start the remote evaluation server

set -e

echo "========================================="
echo "AscendC Remote Evaluation Server Setup"
echo "========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

echo "Python version: $(python3 --version)"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create tasks directory
TASKS_DIR=${TASKS_DIR:-/tmp/ascend_tasks}
mkdir -p "$TASKS_DIR"
echo "Tasks directory: $TASKS_DIR"

# Set environment variables
export TASKS_DIR="$TASKS_DIR"

# Start server
echo ""
echo "Starting server on port 8080..."
echo "Server URL: http://0.0.0.0:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="

python3 app.py
