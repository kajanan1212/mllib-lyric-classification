#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then

    # Check if Python has been installed
    if ! command -v python &> /dev/null; then
        echo "Python is not found. Please install Python and ensure it's added to the PATH."
        exit 1
    fi

    python -m venv venv

fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install project dependencies
pip install --upgrade pip
pip install numpy==1.21.5
pip install pyspark==3.4.1
pip install nltk==3.7
pip install h2o_wave==0.26.3

# Set environment variables
export PYSPARK_PYTHON=venv/bin/python
export PYSPARK_DRIVER_PYTHON=venv/bin/python

# Run the application
wave run src.app

# Deactivate the virtual environment
deactivate
