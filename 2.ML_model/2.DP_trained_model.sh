#!/bin/bash
jupyter nbconvert --to python DP_trained_model.ipynb
python DP_trained_model.py
