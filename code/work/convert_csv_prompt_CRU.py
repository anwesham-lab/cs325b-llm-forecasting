import csv
import json

###################### FILL IN THE RIGHT VALUES HERE  ######################

# output files - SPECIFY
train_json_filename = "../data/CRU/rtrain_lst.jsonl"
test_json_filename = "../data/CRU/rtest_lst.jsonl" 

# input file - SPECIFY
input_csv_filename = '../data/CRU/lst_by_country.csv' 

# system message to "prepare" the llm
prep_msg = "You are a prediction agent predicting the average land surface temperature for a country in a given year when provided an input detailing the country's average land surface temperature for the last 10 years."


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
    input_message = f"The country of interest is: {country_name} and average land surface temperature in degrees Celsius is {i}: {str(round(float(row[str(i)]), 2))}, {i+1}: {str(round(float(row[str(i+1)]), 2))}, {i+2}: {str(round(float(row[str(i+2)]), 2))}, {i+3}: {str(round(float(row[str(i+3)]), 2))}, {i+4}: {str(round(float(row[str(i+4)]), 2))}, {i+5}: {str(round(float(row[str(i+5)]), 2))}, {i+6}: {str(round(float(row[str(i+6)]), 2))}, {i+7}: {str(round(float(row[str(i+7)]), 2))}, {i+8}: {str(round(float(row[str(i+8)]), 2))}, {i+9}: {str(round(float(row[str(i+9)]), 2))}"
    task_message = f". Predict the average land surface temperature (in degrees Celsius) for {country_name} in {prediction_year}: "

    # what the LLM should return
    prediction_amount = str(round(float(row[str(i+predict_after_years)]), 2))
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