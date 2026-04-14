#!/bin/bash
# Test script launcher for CapWorker with TestAgentServer
# 
# This script starts both the TestAgentServer and CapWorker for testing.
# The TestAgentServer simulates an Agent that calls LLM services.
#
# Usage:
#   ./scripts/test_cap_worker.sh [CONFIG_PATH]
#
# Example:
#   ./scripts/test_cap_worker.sh env_configs/cube_stack/franka_robosuite_cube_stack.yaml

set -e

# Default configuration
CONFIG_PATH="${1:-env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml}"
AGENT_URL="ws://localhost:8765/agent"
AUTH_TOKEN=""  # Set this if you want to test authentication

echo "=========================================="
echo "CapWorker Test Setup"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Agent URL: $AGENT_URL"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Starting TestAgentServer..."
echo ""

# Start TestAgentServer in background
uv run --no-sync --active python capx/serving/test_cap_worker.py \
    --config-path "$CONFIG_PATH" \
    --listen-host localhost \
    --listen-port 8765 \
    ${AUTH_TOKEN:+--auth-token "$AUTH_TOKEN"} &

TEST_AGENT_PID=$!
echo "TestAgentServer started (PID: $TEST_AGENT_PID)"

# Wait for server to start
sleep 2

echo ""
echo "Starting CapWorker..."
echo ""

# Start CapWorker (this will connect to TestAgentServer)
uv run --no-sync --active python capx/serving/launch_cap_worker.py \
    --config-path "$CONFIG_PATH" \
    --agent-url "$AGENT_URL" \
    ${AUTH_TOKEN:+--auth-token "$AUTH_TOKEN"} \
    --agent-id "cap-worker-test"

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down TestAgentServer..."
    kill $TEST_AGENT_PID 2>/dev/null || true
    wait $TEST_AGENT_PID 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT
