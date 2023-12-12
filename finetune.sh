#!/bin/bash

echo "Starting Finetuning Job"
python train.py --api_key --file_id --base_model
