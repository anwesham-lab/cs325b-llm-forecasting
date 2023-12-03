import csv
import json

###################### FILL IN THE RIGHT VALUES HERE  ######################

# Train / test splits
train_windows_start = [2001, 2006, 2011] #[1999, 2004, 2009] - for 3y horizon # [2001, 2006, 2011] - for 1y horizon
test_windows_start = [2012]
predict_after_years = 10 # 12 - for 3 year horizon # 1 if "next year"
prediction_horizon = 1 #3

# output files - SPECIFY
train_json_filename = "../data/gdp/train_gdp_growth_" + str(prediction_horizon) + "yr.jsonl" #"../data/gdp/normalized_train_gdp.jsonl" #"../data/gdp/nominal_train_gdp.jsonl"
test_json_filename = "../data/gdp/test_gdp_growth_" + str(prediction_horizon) + "yr.jsonl" #"../data/gdp/test_gdp_growth.jsonl" #"../data/gdp/nominal_test_gdp.jsonl"

# input file - SPECIFY
input_csv_filename = '../data/gdp/rounded_cleaned_gdp_growth_df.csv'
#'../data/gdp/transformed_gdp_df_countries.csv' #'../data/gdp/cleaned_gdp_df_countries.csv'

# system message to "prepare" the llm
growth_prep_msg = "You are a prediction agent predicting the GDP growth rates for a country in a future year when provided an input detailing the country's growth rate in real GDP for the last 10 years."
prep_msg = growth_prep_msg

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
    country_name = row['country'] # loop through each country
    
    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years

    # GDP growth prompts
    input_message = f"The country of interest is: {country_name} and growth rates in percent of real GDP seen in previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    task_message = f". Predict the GDP growth rate (in percent) for {country_name} in {prediction_year}: "

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