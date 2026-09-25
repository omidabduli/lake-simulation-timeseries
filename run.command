#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run.command — One-click launcher for Universal Time-Series Forecasting
# Double-click this file in Finder to start the app.
# ─────────────────────────────────────────────────────────────────────────────

# Move to the script's directory regardless of where Finder launched it from
cd "$(dirname "$0")"

# Make sure libomp (required by XGBoost on macOS) is on the dynamic linker path
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH}"

echo "Starting Universal Time-Series Forecasting..."
echo "   App will open at http://localhost:8502"
echo ""

python3 -m streamlit run app.py --server.port 8502

# Keep terminal window open if there's an error
read -p "Press Enter to close..."
