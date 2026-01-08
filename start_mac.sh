#!/bin/bash
# Activation Steering Research Interface
# Start script for macOS / Linux

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[Activation Steering] Starting...${NC}"

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Clear port 8000 if in use
echo "Checking port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Launch server (use 'info' to see steering debug messages)
echo -e "${GREEN}[Activation Steering] Launching server...${NC}"
echo ""

uvicorn system.server:app --host 0.0.0.0 --port 8000 --log-level info
