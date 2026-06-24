#!/bin/bash

# Test runner script for AI Chatbot Backend
# This script runs all tests with coverage reporting

set -e  # Exit on error

echo "=================================="
echo "AI Chatbot Backend - Test Runner"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Installing test dependencies...${NC}"
    uv pip install pytest pytest-asyncio pytest-mock pytest-cov
fi

echo -e "${YELLOW}Running unit tests...${NC}"
echo ""

# Run unit tests with coverage
uv run pytest tests/unit/\
    -v \
    --tb=short \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml

TEST_EXIT_CODE=$?

echo ""
echo "=================================="

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "Coverage report generated:"
    echo "  - HTML: htmlcov/index.html"
    echo "  - XML: coverage.xml"
    echo ""
    echo "To view HTML coverage report:"
    echo "  open htmlcov/index.html"
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    exit $TEST_EXIT_CODE
fi

echo "=================================="

# Made with Bob
