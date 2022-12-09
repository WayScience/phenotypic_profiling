#!/bin/bash
# Convert notebooks to python file and execute
jupyter nbconvert --to python \
        --FilesWriter.build_directory=scripts/nbconverted \
        --execute *.ipynb
