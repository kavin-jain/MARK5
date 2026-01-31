#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate Virtual Environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual Environment Activated"
else
    echo "⚠️  Virtual Environment not found! Please create one with 'python3 -m venv .venv'"
fi

# Kill any existing processes on ports 8000 (API) and 3000 (UI)
fuser -k 8000/tcp 2>/dev/null
fuser -k 3000/tcp 2>/dev/null

echo "🚀 Starting MARK3 System..."

# Start Backend
echo "Starting API Server..."
python3 -m uvicorn core.api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start Frontend
# Start Frontend
# echo "Starting Web UI..."
# cd mark3-ui
# npm run dev &
# UI_PID=$!

echo "✅ System Online!"
echo "API: http://localhost:8000"
echo "UI:  http://localhost:3000"
echo "Press Ctrl+C to stop."

# Wait for processes
wait $API_PID $UI_PID
