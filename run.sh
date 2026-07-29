#!/bin/bash
# run.sh

echo "Starting Causal Inference Application..."

# Start FastAPI backend in background
echo "Starting FastAPI backend..."
uvicorn app.api.endpoints:app --host 0.0.0.0 --port 8000 --reload &

# Wait for backend to start
sleep 5

# Start Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# Kill background processes on exit
trap 'kill $(jobs -p)' EXIT
wait