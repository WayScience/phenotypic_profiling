#!/bin/bash
# Convert notebook to python file and execute
jupyter nbconvert --to python \
        --output-dir=scripts/nbconverted \
        --execute correct_15_images.ipynb
