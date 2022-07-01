#!/bin/bash
jupyter nbconvert --to python format_training_data.ipynb
python format_training_data.py
