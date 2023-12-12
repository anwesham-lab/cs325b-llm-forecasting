#!/bin/bash

echo "Converting csv to jsonl"
python gen_jsonl.py --n --global_start_year --global_end_year --features --n_train_rows --system_content --train_windows --test_window --train_filename --test_filename --csv_file

echo "Uploading Finetuning Job"
python upload.py --api_key --train_filename
