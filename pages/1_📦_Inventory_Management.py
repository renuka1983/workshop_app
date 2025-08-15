import streamlit as st
import sys
import os

# Add the inventory directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inventory'))

# Import and run the inventory management app
from inventory_management import main

if __name__ == "__main__":
    main()
