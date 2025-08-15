import streamlit as st
import sys
import os

# Add the predictive_maintenance directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'predictive_maintenance'))

# Import and run the predictive maintenance app
from app import main

if __name__ == "__main__":
    main()

