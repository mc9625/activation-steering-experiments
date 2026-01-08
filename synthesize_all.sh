#!/bin/bash
# Synthesize all compounds from JSON definitions
# This script requires the model to be downloaded first (see README)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}COMPOUND SYNTHESIS${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Check HuggingFace authentication
echo ""
echo "Checking HuggingFace authentication..."
if ! python3 -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>/dev/null; then
    echo -e "${YELLOW}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  WARNING: Not logged in to HuggingFace!                  ║"
    echo "║                                                          ║"
    echo "║  Llama 3.2 is a gated model. You need to:               ║"
    echo "║  1. Create account: https://huggingface.co/join         ║"
    echo "║  2. Request access to Llama 3.2 (link in README)        ║"
    echo "║  3. Run: huggingface-cli login                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please login first with: huggingface-cli login"
        exit 1
    fi
else
    echo -e "${GREEN}✓ HuggingFace authentication OK${NC}"
fi

# Check for substance definitions
SUBSTANCES_DIR="substances"
if [ ! -d "$SUBSTANCES_DIR" ]; then
    echo -e "${RED}[ERROR] substances/ directory not found${NC}"
    exit 1
fi

# Count JSON files
JSON_COUNT=$(ls -1 $SUBSTANCES_DIR/*.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo -e "${RED}[ERROR] No JSON files found in substances/${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Found $JSON_COUNT compound definitions${NC}"
echo ""

# Synthesize each compound
for json_file in $SUBSTANCES_DIR/*.json; do
    name=$(basename "$json_file" .json)
    echo -e "${YELLOW}Synthesizing: $name${NC}"
    
    python tools/synthesize.py --file "$json_file" --outdir vectors
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name complete${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
    fi
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SYNTHESIS COMPLETE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Vectors saved to: vectors/"
ls -la vectors/*.pt 2>/dev/null || echo "(no vectors found)"
