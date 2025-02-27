#!/bin/bash

# Start the Python subprocess
python main.py &
PYTHON_PID=$!

# Function to clean up the Python subprocess and its children
cleanup() {
    echo "Cleaning up subprocess (PID: $PYTHON_PID) and its children..."
    # Find all child processes of the Python process and kill them
    pkill -TERM -P $PYTHON_PID
    sleep 1
    kill -9 $PYTHON_PID 2>/dev/null
}

# Function to handle signals
handle_signal() {
    SIGNAL=$1
    echo "Received signal $SIGNAL. Resending to Python subprocess..."
    kill -$SIGNAL $PYTHON_PID
}

# Trap SIGINT and SIGQUIT signals
trap 'handle_signal SIGINT' SIGINT
trap 'handle_signal SIGQUIT' SIGQUIT

# Timeout duration (in seconds)
TIMEOUT=10

# Wait for the subprocess or timeout
(
    sleep $TIMEOUT
    echo "Timeout reached. Killing Python subprocess..."
    cleanup
) &
TIMEOUT_PID=$!

# Wait for the Python subprocess to exit
wait $PYTHON_PID
EXIT_CODE=$?

# Kill the timeout watcher if the subprocess finishes first
kill $TIMEOUT_PID 2>/dev/null

# Final cleanup
cleanup

# Exit with the Python subprocess's exit code
exit $EXIT_CODE
