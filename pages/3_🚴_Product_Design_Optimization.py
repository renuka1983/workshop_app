import streamlit as st
import sys
import os

# Add the product_design directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'product_design'))

# Import and run the design optimization app
from design_optimization import main

if __name__ == "__main__":
    main()
