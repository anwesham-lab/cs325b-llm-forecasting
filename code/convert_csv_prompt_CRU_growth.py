import csv
import json

###################### FILL IN THE RIGHT VALUES HERE  ######################

# output files - SPECIFY
train_json_filename = "../data/CRU/train_lst_growth.jsonl"
test_json_filename = "../data/CRU/test_lst_growth.jsonl" 

# input file - SPECIFY
input_csv_filename = '../data/CRU/lst_growth_by_country.csv' 

# system message to "prepare" the llm
prep_msg = "You are a prediction agent predicting the average land surface temperature growth rate for a country in a given year when provided an input detailing the country's average land surface temperature growth rate for the last 10 years."


# Train / test splits
train_windows_start = [1902] + [x for x in range(1911, 2012, 5)] 
test_windows_start = [2012]
predict_after_years = 10

############################################################################

# Read the CSV file 
with open(input_csv_filename, newline='') as csvfile:
    data = list(csv.DictReader(csvfile))

# Define the system message
system_message = prep_msg


# Initialize a list to store the prompts
train_prompts_json = []
test_prompts_json = []

##
## helper function to create train and test windows
##

def generate_message(row, window_start):
    country_name = row['Country'] # loop through each country
    
    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years

    # # regular GDP prompts
    input_message = f"The country of interest is: {country_name} and average land surface temperature growth rate in degrees Celsius is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    task_message = f". Predict the average land surface temperature growth rate (in degrees Celsius) for {country_name} in {prediction_year}: "

    # what the LLM should return
    prediction_amount = row[str(i+predict_after_years)]
    agent_message = f"{prediction_amount}"

    # Append the input and agent messages to the prompts list
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_message + task_message},
            {"role": "assistant", "content": agent_message}
        ]
    }

##
## training jsonl create and write
##        

# Loop through the data
for w in train_windows_start:
    for row in data:
        # Extract relevant data for that country and start year combination
        train_prompts_json.append(generate_message(row, w))
len_content = len(train_prompts_json)

# Save the prompts as a JSON file    
with open(train_json_filename, 'w') as f:
    for entry in train_prompts_json:
        f.write(json.dumps(entry) + '\n')

##
## test jsonl create and write
##        

# Loop through the data
for w in test_windows_start:
    for row in data:
        # Extract relevant data
        test_prompts_json.append(generate_message(row, w))
len_content = len(test_prompts_json)

# Save the prompts as a JSON file    
with open(test_json_filename, 'w') as f:
    for entry in test_prompts_json:
        f.write(json.dumps(entry) + '\n')