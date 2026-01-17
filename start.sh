#!/bin/bash

# Configuration
VENV_PATH="env/bin/activate"
SERVER_PORT=8000
UI_PORT=8501

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if environment exists
if [ ! -f "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Please create it or update script path."
    exit 1
fi

echo -e "${GREEN}Activating Environment...${NC}"
source $VENV_PATH

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${RED}Stopping services...${NC}"
    kill $SERVER_PID
    kill $UI_PID
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo -e "${GREEN}Starting Backend Server at http://localhost:$SERVER_PORT...${NC}"
uvicorn server.main:app --port $SERVER_PORT &
SERVER_PID=$!

# Wait a bit for server to start
sleep 3

echo -e "${GREEN}Starting SmartVision UI at http://localhost:$UI_PORT...${NC}"
streamlit run pipeline/app.py --server.port $UI_PORT &
UI_PID=$!

echo -e "${GREEN}System Running. Press Ctrl+C to stop.${NC}"

# Keep script running
wait
