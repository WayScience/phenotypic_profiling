#!/bin/bash
jupyter nbconvert --to python 0.split_data.ipynb
python 0.split_data.py
jupyter nbconvert --to python 1.train_model.ipynb
python 1.train_model.py
jupyter nbconvert --to python 2.evaluate_model.ipynb
python 2.evaluate_model.py
jupyter nbconvert --to python 3.interpret_model.ipynb
python 3.interpret_model.py
