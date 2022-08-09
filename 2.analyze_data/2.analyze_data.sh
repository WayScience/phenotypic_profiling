#!/bin/bash
# Step 0: Convert notebook to script
jupyter nbconvert --to=script analyze_training_data.ipynb

# Step 1: Execute jupyter notebook
jupyter nbconvert --to=html \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=10000000 \
        --execute analyze_training_data.ipynb
