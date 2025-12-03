#!/bin/bash
set -e

echo "Starting Alaska Snow Agent..."
echo ""

# Run resource validation
python validate_resources.py

# If validation passes, start Streamlit
if [ $? -eq 0 ]; then
    exec streamlit run app.py --server.port=8080 --server.address=0.0.0.0
else
    echo ""
    echo "❌ Startup failed: Required resources not found"
    exit 1
fi
