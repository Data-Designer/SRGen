#!/bin/bash

# TNOT Model Server Startup Script

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CONFIG_FILE="${CONFIG_FILE:-server_config.json}"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TNOT Model Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Warning: Config file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Using command-line arguments or defaults${NC}"
    echo ""
fi

# Display configuration
echo -e "${GREEN}Server Configuration:${NC}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo -e "  Config File: ${CONFIG_FILE}"
echo ""

# Check for Python and required packages
echo -e "${GREEN}Checking dependencies...${NC}"
python3 -c "import fastapi, uvicorn, transformers, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some required packages may be missing${NC}"
    echo -e "${YELLOW}Install with: pip install fastapi uvicorn transformers torch${NC}"
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting server...${NC}"
echo ""

if [ -f "$CONFIG_FILE" ]; then
    python3 srgen_server.py \
        --config "$CONFIG_FILE" \
        --host "$HOST" \
        --port "$PORT" \
        "$@"
else
    python3 srgen_server.py \
        --host "$HOST" \
        --port "$PORT" \
        "$@"
fi
