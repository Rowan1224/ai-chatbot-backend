#!/bin/bash
# interactive_test.sh — interactive API test REPL
# Run from anywhere: scripts/interactive_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Interactive API Test Script for AI Chatbot Backend
# Allows you to chat with the bot interactively

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-dev-api-key-12345}"
SESSION_ID="interactive-session-$(date +%s)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================="
echo "AI Chatbot Interactive Test"
echo "==========================================${NC}"
echo ""
echo "API URL: $API_URL"
echo "Session ID: $SESSION_ID"
echo ""
echo -e "${BLUE}Instructions:${NC}"
echo "- Type your messages and press Enter"
echo "- Type 'quit' or 'exit' to end the session"
echo "- Type 'state' to see current session state"
echo "- Type 'health' to check API health"
echo "- The session exits automatically when the conversation completes"
echo ""
echo -e "${YELLOW}Conversation Flow:${NC}"
echo "1. Describe your request (e.g., 'I need infrastructure provisioning')"
echo "2. Provide details when asked"
echo "3. When asked for PII, use format: NAME,EMPLOYEE_ID"
echo "   Example: John Doe,EMP12345"
echo "4. Confirm with 'yes' or provide corrections"
echo ""
echo "=========================================="
echo ""

# Function to send message
# Returns exit code 2 when the conversation is complete so the
# caller can break out of the loop.
send_message() {
    local message="$1"

    response=$(curl -s -X POST "$API_URL/chat" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d "{\"session_id\": \"$SESSION_ID\", \"message\": \"$message\"}" 2>&1)

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to connect to API${NC}"
        echo "Make sure the server is running: uvicorn src.api.main:app --reload"
        return 1
    fi

    # Extract fields from response
    bot_response=$(echo "$response" | jq -r '.response' 2>/dev/null)
    is_complete=$(echo "$response" | jq -r '.is_complete' 2>/dev/null)
    request_active=$(echo "$response" | jq -r '.request_active' 2>/dev/null)
    duplicate_warning=$(echo "$response" | jq -r '.duplicate_warning' 2>/dev/null)

    if [ "$bot_response" = "null" ] || [ -z "$bot_response" ]; then
        echo -e "${RED}Error: Invalid response from API${NC}"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
        return 1
    fi

    echo -e "${GREEN}Bot:${NC} $bot_response"

    # Show duplicate warning if present
    if [ "$duplicate_warning" != "null" ] && [ "$duplicate_warning" != "[]" ]; then
        echo -e "${YELLOW}⚠️  Duplicates found!${NC}"
        echo "$response" | jq '.duplicate_warning'
    fi

    # Show completion status and signal the loop to exit
    if [ "$is_complete" = "true" ] && [ "$request_active" = "false" ]; then
        echo -e "${GREEN}✅ Conversation complete!${NC}"
        echo ""
        echo "Final data:"
        echo "$response" | jq '.collected_data'
        echo ""
        return 2  # sentinel: conversation finished
    fi
    if [ "$is_complete" = "false" ] && [ "$request_active" = "false" ]; then
        echo -e "${YELLOW}Conversation ended.${NC}"
        echo "User ended the conversation"
        return 2  # sentinel: conversation finished
    fi

    echo ""
}

# Function to check health
check_health() {
    echo -e "${BLUE}Checking API health...${NC}"
    curl -s "$API_URL/health" | jq '.'
    echo ""
}

# Function to show state
show_state() {
    echo -e "${BLUE}Current session state:${NC}"
    curl -s "$API_URL/session/$SESSION_ID" \
        -H "X-API-Key: $API_KEY" | jq '.'
    echo ""
}

# Initial health check
check_health

# Main loop
while true; do
    echo -ne "${BLUE}You:${NC} "
    read -r user_input
    
    # Check for special commands
    case "$user_input" in
        quit|exit)
            echo "Goodbye!"
            exit 0
            ;;
        state)
            show_state
            continue
            ;;
        health)
            check_health
            continue
            ;;
        "")
            continue
            ;;
    esac
    
    # Send message to API
    send_message "$user_input"
    rc=$?
    if [ $rc -eq 2 ]; then
        # Conversation completed — exit automatically
        exit 0
    fi
done


