#!/bin/bash

# Timeout duration (in seconds)
TIMEOUT=5

# Make preparations
if [ -d ../venv ] 
then
  . ../venv/bin/activate
else 
  . ./venv/bin/activate
fi

# Limit number of threads numpy uses:
export OPENBLAS_NUM_THREADS=4
export GOTO_NUM_THREADS=4
export OMP_NUM_THREADS=4

DIR=$(dirname $0)
export PYTHONPATH=.

# Start the Python subprocess
python ${DIR}/main.py $@ &
PYTHON_PID=$!

# Function to clean up the Python subprocess and its children
cleanup() {
    echo "Cleaning up subprocess (PID: $PYTHON_PID) and its children..."
    # Find all child processes of the Python process and kill them
    pkill -TERM -P $PYTHON_PID
    sleep 1
    pkill -9 -P $PYTHON_PID
    kill -9 $PYTHON_PID 2>/dev/null
}

# Function to handle signals
handle_signal() {
    SIGNAL=$1
    echo "Received signal $SIGNAL. Resending to Python subprocess..."
    kill -$SIGNAL $PYTHON_PID

    # Wait for the subprocess or timeout
    (
        sleep $TIMEOUT
        echo "Timeout reached. Killing Python subprocess..."
        cleanup
    ) &
    TIMEOUT_PID=$!
    # Kill the timeout watcher after 2 TIMEOUTs
    (
        sleep $TIMEOUT
        sleep $TIMEOUT
        echo "Timeout reached. Killing timeout watcher..."
        kill $TIMEOUT_PID 2>/dev/null
    ) &
}

# Trap SIGINT and SIGQUIT signals
trap 'handle_signal SIGINT' SIGINT
trap 'handle_signal SIGTERM' SIGTERM
trap 'handle_signal SIGQUIT' SIGQUIT


# Wait for the Python subprocess to exit
wait $PYTHON_PID
EXIT_CODE=$?


# Final cleanup
# cleanup # !!! Note: Doesn't make sense, main process is already dead, children is alive have 1 as ppid

# Exit with the Python subprocess's exit code
exit $EXIT_CODE
