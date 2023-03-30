#!/bin/bash
# Convert notebook to python file and execute
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute train_multi_class_models.ipynb

jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute train_single_class_models.ipynb
