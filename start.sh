#!/bin/bash

PYTHON=/Users/xiexiaoxuan/opt/miniconda3/envs/tf_py3/bin/python
$PYTHON comm.py
$PYTHON baseline.py offline_train
$PYTHON baseline.py evaluate
$PYTHON baseline.py online_train
$PYTHON baseline.py submit
$PYTHON evaluation.py

