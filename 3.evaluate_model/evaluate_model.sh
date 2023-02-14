#!/bin/bash
# Convert notebooks to python file and execute
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute get_model_predictions.ipynb

jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute class_PR_curves.ipynb

jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute confusion_matrices.ipynb

jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute F1_scores.ipynb
