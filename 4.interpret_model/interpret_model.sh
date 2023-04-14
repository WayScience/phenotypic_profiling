#!/bin/bash
# Convert notebook to python file and execute
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute interpret_model_coefficients.ipynb

jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute interpret_SCM_model_coefficients.ipynb

jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute compare_model_coefficients.ipynb
