#!/bin/bash

# 🔥 MARK3 Trading Terminal Launcher

cd "$(dirname "$0")"

echo "================================================"
echo "🔥 MARK3 PROFESSIONAL TRADING TERMINAL"
echo "================================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Create logs directory
mkdir -p logs

echo "✅ Environment activated"
echo "🚀 Launching professional terminal..."
echo ""
echo "📊 Launching Terminal Interface..."
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch Terminal UI
python3 core/terminal_ui_enhanced.py
