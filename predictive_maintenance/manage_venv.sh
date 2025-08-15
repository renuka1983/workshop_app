#!/bin/bash

# Virtual Environment Management Script for Predictive Maintenance App

VENV_DIR="venv"

case "$1" in
    "activate")
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
        echo "Virtual environment activated! Use 'deactivate' to exit."
        echo "Current Python: $(which python)"
        echo "Current pip: $(which pip)"
        ;;
    "deactivate")
        echo "Deactivating virtual environment..."
        deactivate
        ;;
    "install")
        echo "Installing dependencies..."
        source "$VENV_DIR/bin/activate"
        pip install -r requirements.txt
        ;;
    "run-app")
        echo "Starting Streamlit app..."
        source "$VENV_DIR/bin/activate"
        streamlit run app.py --server.headless true --server.port 8501
        ;;
    "run-demo")
        echo "Running demo script..."
        source "$VENV_DIR/bin/activate"
        python demo.py
        ;;
    "clean")
        echo "Removing virtual environment..."
        rm -rf "$VENV_DIR"
        echo "Virtual environment removed."
        ;;
    "recreate")
        echo "Recreating virtual environment..."
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install -r requirements.txt
        echo "Virtual environment recreated and dependencies installed."
        ;;
    *)
        echo "Usage: $0 {activate|deactivate|install|run-app|run-demo|clean|recreate}"
        echo ""
        echo "Commands:"
        echo "  activate    - Activate the virtual environment"
        echo "  deactivate  - Deactivate the virtual environment"
        echo "  install     - Install dependencies in activated venv"
        echo "  run-app     - Start the Streamlit app"
        echo "  run-demo    - Run the demo script"
        echo "  clean       - Remove the virtual environment"
        echo "  recreate    - Remove and recreate the virtual environment"
        echo ""
        echo "Example:"
        echo "  source manage_venv.sh activate  # Activate venv"
        echo "  ./manage_venv.sh run-app        # Start app"
        echo "  ./manage_venv.sh run-demo       # Run demo"
        ;;
esac 