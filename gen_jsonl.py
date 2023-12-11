import argparse
import json
import pandas as pd

def get_args_parser():
    parser = argparse.ArgumentParser('Generate a jsonl file of messages from a csv')
    parser.add_argument("--n", type=int, help="Number of rows (train+test) to subsample from")
    parser.add_argument("--global_start_year", type=int, help="Earliest start year from first training window")
    parser.add_argument("--global_end_year", type=int, help="Last year, most likely prediction test year")
    parser.add_argument("--features", type=list_of_strings, help="Column names/metadata to include separated by commas")
    parser.add_argument("--n_train_rows", type=int, help="Number of rows for training (note that training samples will = number of training windows * number of rows for training)")
    parser.add_argument("--system_content", type=str, help="Content for system role (string)")
    parser.add_argument("--train_windows", type=list_of_tuples, help="Pairs of start and end year (inclusive) of a window, separated by commas")
    parser.add_argument("--test_window", type=single_tuple, help="A start and end year (inclusive) of a window, separated by commas")
    parser.add_argument("--train_filename", type=str, help="Filename to write the training dataset to")
    parser.add_argument("--test_filename", type=str, help="Filename to write the testing dataset to")
    parser.add_argument("--csv_file", type=str, help="Filename of csv file")

    return parser

def list_of_strings(arg):
    return arg.split(',')

def single_tuple(arg):
    return tuple([int(a) for a in arg.split(',')])

def list_of_tuples(arg):
    years = arg.split(',')
    return [tuple([int(years[i-1]), int(years[i])]) for i in range(1, len(years), 2)]

def generate_user_content(row, features, start_year, end_year):
    user_content = ', '.join([f"{feature}: {row[feature]}" for feature in features])
    user_content += '. '
    content = []
    for i in range(start_year + 1, end_year+1):
        prev_val, cur_val = row[str(i-1)], row[str(i)]
        val = (cur_val - prev_val)/prev_val*100
        val = round(val, 2)
        content.append(str(i-1) + '-' + str(i) + ' = ' + str(val))
    user_content += ', '.join(content)
    return user_content

def generate_assistant_content(row, end_year):
    val = (row[str(end_year + 1)] - row[str(end_year)])/row[str(end_year)]*100
    val = round(val, 2)
    assistant_content = str(end_year) + '-' + str(end_year+1) + ' = ' + str(val)
    return assistant_content

def generate_message(row, features, start_year, end_year, system_content):
    user_content = generate_user_content(row, features, start_year, end_year)
    assistant_content = generate_assistant_content(row, end_year)
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def generate_train_messages(row, features, system_content, windows):
    train_messages = []
    for window in windows:
        message = generate_message(row, features, window[0], window[1], system_content)
        train_messages.append(message)
    return train_messages

def generate_test_message(row, features, system_content, window):
    test_message = generate_message(row, features, window[0], window[1], system_content)
    return test_message

def contains_nan(row, start_year, end_year, features):
    years = [str(year) for year in range(start_year, end_year + 1)]  
    for col in years + features:
        if pd.isna(row[col]):
            return True
    return False

def process_csv(filename, n, global_start_year, global_end_year, features):
    df = pd.read_csv(filename)
    filtered_df = df[~df.apply(contains_nan, args=(global_start_year, global_end_year, features), axis=1)]

    subset_df = filtered_df.sample(n=n)
    return subset_df

def generate_datasets(df, n_train_rows, features, system_content, train_windows, test_window):
    train=[]
    test=[]
    counter = 0
    for _, row in df.iterrows():
        if counter < n_train_rows:
            train_messages = generate_train_messages(row, features, system_content, train_windows)
            train.extend(train_messages)
        else:
            test_message = generate_test_message(row, features, system_content, test_window)
            test.append(test_message)
        counter += 1
    return train, test

def generate_jsonl(train_filename, train_data, test_filename, test_data):
    with open(train_filename, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    with open(test_filename, 'w') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
        
def main(args):
    df = process_csv(args.csv_file, args.n, args.global_start_year, args.global_end_year, args.features)
    train_data, test_data = generate_datasets(df, args.n_train_rows, args.features, args.system_content, args.train_windows, args.test_window)
    generate_jsonl(args.train_filename, train_data, args.test_filename, test_data)

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
