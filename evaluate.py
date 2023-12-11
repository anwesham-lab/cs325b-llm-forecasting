import argparse
import json
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def get_args_parser():
    parser = argparse.ArgumentParser('Perform inference via finetuned model')
    parser.add_argument("--results_filename", type=str, help="Inference results file")
    parser.add_argument("--metric", type=str, help="mae or mape")
    return parser


def read_json(file_path):
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    return loaded_data

def evaluate(gt, pred, metric):
    print(f"MAE is {mean_absolute_error(gt, pred)}") if metric == "mae" else print(f"MAPE is {mean_absolute_percentage_error(gt, pred)}")

def main(args):
    results = read_json(args.results_filename)
    evaluate(results["gt"], results["pred"], args.metric)

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
