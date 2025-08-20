#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical dependencies
python -c "import dateutil, spacy, tensorflow, sklearn"

# Use gunicorn directly with config
python -m spacy download en_core_web_sm  # Add this line
gunicorn app:app --config gunicorn.conf.py