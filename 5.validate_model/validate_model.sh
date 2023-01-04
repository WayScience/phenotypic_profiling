#!/bin/bash
# Convert notebook to python file and execute
jupyter nbconvert --to python \
        --output-dir=scripts/nbconverted \
        --execute cell_health_validation.ipynb
