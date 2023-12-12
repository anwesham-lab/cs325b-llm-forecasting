#!/bin/bash

echo "Starting inference"
python inference.py --api_key --model_id --results_filename --test_filename

echo "Starting evaluation"
python evaluate.py --results_filename --metric
