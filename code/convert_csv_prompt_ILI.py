import csv
import json

###################### FILL IN THE RIGHT VALUES HERE  ######################

# output files - SPECIFY
train_json_filename = "../data/ILI/trainili.jsonl" #"../data/gdp/nominal_train_gdp.jsonl"
test_json_filename = "../data/ILI/testili.jsonl" #"../data/gdp/nominal_test_gdp.jsonl"

# input file - SPECIFY
input_csv_filename = '../data/ILI/cleaned_ili.csv' #'../data/gdp/cleaned_gdp_df_countries.csv'

# system message to "prepare" the llm
nom_prep_msg = "You are a prediction agent predicting the number of patients seen with influenza-Like illness for an age demographic in a given year when provided an input detailing the number of new cases for the last 10 years."
prep_msg = nom_prep_msg


# Train / test splits
train_windows_start = [1998, 2000, 2002, 2004, 2006, 2008, 2010] # [2001, 2006, 2011] - for nominal
test_windows_start = [2011, 2012]
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
    demographic = str.lower(row['Demographic']) # loop through each country
    if demographic == 'ilitotal':
        demographic = 'all ages'
    
    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years

    # # regular GDP prompts
    input_message = f"The demographic of interest is: {demographic} and the number of pateints seen with influenzalike illness for this demographic in previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    task_message = f". Predict the number of patients for {demographic} in {prediction_year}: "

    # normalized GDP prompts
    # input_message = f"We have normalized GDP values to be between 0 and 10. The country of interest is: {country_name} and normalized GDP from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    # task_message = f". Predict the normalized GDP for {country_name} in {prediction_year}: "

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