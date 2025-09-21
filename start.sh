#!/bin/bash
# Gold Price Forecasting System - Quick Start Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/Users/hasannumanoglu/Documents/SoftDev/GoldPriceForecasting"
PYTHON_CMD="$PROJECT_DIR/.venv/bin/python"

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}    GOLD PRICE FORECASTING SYSTEM${NC}"
echo -e "${GREEN}===============================================${NC}"
echo

# Check if virtual environment exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run the setup first."
    exit 1
fi

# Menu
echo -e "${BLUE}Choose an option:${NC}"
echo "1. Test basic functionality"
echo "2. Run full forecasting system (main.py)"
echo "3. Start API server"
echo "4. Open Jupyter notebook"
echo "5. Run tests"
echo

read -p "Enter your choice (1-5): " choice

cd "$PROJECT_DIR"

case $choice in
    1)
        echo -e "${YELLOW}Running basic functionality test...${NC}"
        $PYTHON_CMD test_basic.py
        ;;
    2)
        echo -e "${YELLOW}Running full forecasting system...${NC}"
        $PYTHON_CMD main.py
        ;;
    3)
        echo -e "${YELLOW}Starting API server...${NC}"
        echo "Installing missing dependencies..."
        $PYTHON_CMD -m pip install tensorflow keras > /dev/null 2>&1
        echo "API will be available at: http://localhost:8000"
        echo "API docs will be available at: http://localhost:8000/docs"
        echo "Press Ctrl+C to stop the server"
        $PYTHON_CMD -m uvicorn api.fastapi_app:app --reload --host 127.0.0.1 --port 8000
        ;;
    4)
        echo -e "${YELLOW}Opening Jupyter notebook...${NC}"
        echo "Notebook will open in your browser"
        $PYTHON_CMD -m pip install jupyter > /dev/null 2>&1
        $PYTHON_CMD -m jupyter notebook notebooks/exploration.ipynb
        ;;
    5)
        echo -e "${YELLOW}Running tests...${NC}"
        $PYTHON_CMD -m pytest tests/ -v
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac