import csv
import json

###################### FILL IN THE RIGHT VALUES HERE  ######################

# output files - SPECIFY
train_json_filename = "../data/gdp/normalized_train_gdp.jsonl" #"../data/gdp/nominal_train_gdp.jsonl"
test_json_filename = "../data/gdp/normalized_test_gdp.jsonl" #"../data/gdp/nominal_test_gdp.jsonl"

# input file - SPECIFY
input_csv_filename = '../data/gdp/transformed_gdp_df_countries.csv' #'../data/gdp/cleaned_gdp_df_countries.csv'

# system message to "prepare" the llm
nom_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years."
norm_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years. The GDP has been normalized so that it is always between 0 and 10."
prep_msg = norm_prep_msg


# Train / test splits
train_windows_start = [2002, 2006, 2011] # [2001, 2006, 2011] - for nominal
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
    country_name = row['country'] # loop through each country
    
    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years

    # # regular GDP prompts
    # input_message = f"The country of interest is: {country_name} and GDP (in Billions of US dollars at present day prices) from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    # task_message = f". Predict the GDP (in Billions of US dollars at present day prices) for {country_name} in {prediction_year}: "

    # normalized GDP prompts
    input_message = f"We have normalized GDP values to be between 0 and 10. The country of interest is: {country_name} and normalized GDP from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    task_message = f". Predict the normalized GDP for {country_name} in {prediction_year}: "

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