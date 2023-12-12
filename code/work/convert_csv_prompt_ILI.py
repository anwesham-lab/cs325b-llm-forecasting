import csv
import json

###################### FILL IN THE RIGHT VALUES HERE  ######################

# output files - SPECIFY
train_json_filename = "../data/ILI/trainilim.jsonl" #"../data/gdp/nominal_train_gdp.jsonl"
test_json_filename = "../data/ILI/testilim.jsonl" #"../data/gdp/nominal_test_gdp.jsonl"

# input file - SPECIFY
input_csv_filename = '../data/ILI/cleaned_monthly_ILI.csv' #'../data/gdp/cleaned_gdp_df_countries.csv'

# system message to "prepare" the llm
nom_prep_msg = "You are a prediction agent predicting the number of new patients seen with influenza-Like illness in the United States for an age demographic in a given month of a given year when provided an input detailing the number of new cases for the last 10 years in that same month."
prep_msg = nom_prep_msg


# Train / test splits
train_windows_start = [1998, 2001, 2003, 2005, 2007, 2009, 2011] # [2001, 2006, 2011] - for nominal
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

def generate_message(row, window_start, month):
    demographic = str.lower(row['Demographic']) # loop through each country
    if demographic == 'ilitotal':
        demographic = 'all ages'

    month_dict = {
        1:'January',
        2:'February',
        3:'March',
        4:'April',
        5:'May',
        6:'June',
        7:'July',
        8:'August',
        9:'September',
        10:'October',
        11:'November',
        12:'December'		
    }
    
    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years
    j = month

    # # regular GDP prompts
    input_message = f"The demographic of interest is: {demographic} for the month of {month_dict[j]} and the number of new patients seen with influenzalike illness for this demographic in previous years is {i}: {row[str(i) + '-' + str(j)]}, {i+1}: {row[str(i+1)+ '-' + str(j)]}, {i+2}: {row[str(i+2)+ '-' + str(j)]}, {i+3}: {row[str(i+3)+ '-' + str(j)]}, {i+4}: {row[str(i+4)+ '-' + str(j)]}, {i+5}: {row[str(i+5)+ '-' + str(j)]}, {i+6}: {row[str(i+6)+ '-' + str(j)]}, {i+7}: {row[str(i+7)+ '-' + str(j)]}, {i+8}: {row[str(i+8)+ '-' + str(j)]}, {i+9}: {row[str(i+9)+ '-' + str(j)]}"
    task_message = f". Predict the number of new influenzalike illness patients in {month_dict[j]} for {demographic} in {prediction_year}: "

    # normalized GDP prompts
    # input_message = f"We have normalized GDP values to be between 0 and 10. The country of interest is: {country_name} and normalized GDP from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    # task_message = f". Predict the normalized GDP for {country_name} in {prediction_year}: "

    # what the LLM should return
    prediction_amount = row[str(i+predict_after_years)+ '-' + str(j)]
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
        for j in range(1, 13):
        # Extract relevant data for that country and start year combination
            train_prompts_json.append(generate_message(row, w, j))
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
        for j in range(1, 11):
        # Extract relevant data
            test_prompts_json.append(generate_message(row, w, j))
len_content = len(test_prompts_json)

# Save the prompts as a JSON file    
with open(test_json_filename, 'w') as f:
    for entry in test_prompts_json:
        f.write(json.dumps(entry) + '\n')