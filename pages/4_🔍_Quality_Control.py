import streamlit as st
import sys
import os

# Add the QualityControl directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'QualityControl'))

# Import and run the defect detection app
from defect_detection import main

if __name__ == "__main__":
    main()
