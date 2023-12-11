import argparse
import openai

def get_args_parser():
    parser = argparse.ArgumentParser('Finetune via OpenAI API')
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--file_id", type=str, help="file id for finetuning (from upload response)")
    parser.add_argument("--base_model", type=str, default="gpt-3.5-turbo", help="Base OpenAI model")
    return parser

def finetune(file_id, base_model):
    response = openai.FineTuningJob.create(training_file=file_id, model=base_model)
    print(response)

def main(args):
    openai.api_key = args.api_key
    finetune(args.file_id, args.base_model)


if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)