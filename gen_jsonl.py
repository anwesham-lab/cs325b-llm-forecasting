import json
import pandas as pd


def generate_user_content(row, features, start_year, end_year):
    user_content = ', '.join([f"{feature}: {row[feature]}" for feature in features])
    user_content += '. '
    user_content += ', '.join([f"{year}={row[str(year)]}" for year in range(start_year, end_year + 1)])
    return user_content

def generate_assistant_content(row, end_year):
    assistant_content = str(end_year + 1) + '=' + str(row[str(end_year + 1)])
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

def generate_datasets(df, train_split, features, train_system_content, test_system_content, train_windows, test_window):
    n_train_rows = train_split*df.shape[0]/len(train_windows)
    train=[]
    test=[]
    counter = 0
    for _, row in df.iterrows():
        if counter < n_train_rows:
            train_messages = generate_train_messages(row, features, train_system_content, train_windows)
            train.extend(train_messages)
        else:
            test_message = generate_test_message(row, features, test_system_content, test_window)
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
        
def main():
    # csv file
    filename = ".csv"
    # number of rows to sample dataset from 
    n = 
    # earliest start year from first window
    global_start_year = 
    # prediction year for testing (most likely 2022)
    global_end_year = 
    # column names/metadata to include as a list of strings
    features = ['','']
    # proportion of split that is training data (Float from 0 to 1)
    train_split = 
    # content for system role for training (string)
    train_system_content = ""
    # content for system role for test (string)
    test_system_content = ""
    # List of tuples/lists, where each tuple contains a start and end year (inclusive) of a window
    train_windows = [(,),(,)]
    # Tuple/List, containing a start and end year (inclusive) of a window
    test_window = (,)
    # filename to write the training dataset to
    train_filename = ".jsonl"
    # filename to write the test dataset to
    test_filename = ".jsonl"
    df = process_csv(filename, n, global_start_year, global_end_year, features)
    train_data, test_data = generate_datasets(df, train_split, features, train_system_content, test_system_content, train_windows, test_window)
    generate_jsonl(train_filename, train_data, test_filename, test_data)

if __name__=="__main__":
    main()
