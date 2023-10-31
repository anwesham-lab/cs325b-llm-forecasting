import pandas as pd
import csv
import json
import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

test_year = "2022"

###################### FILL IN THE RIGHT VALUES HERE  ######################

# input file - SPECIFY
input_csv_filename = "../data/gdp/test_cleaned_gdp.csv" #'../data/gdp/cleaned_gdp_df_countries.csv'

# system message to "prepare" the llm
nom_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years."
norm_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years. The GDP has been normalized so that it is always between 0 and 10."
prep_msg = nom_prep_msg

# Train / test splits
test_windows_start = 2012
predict_after_years = 10

# limit how many predictions to make
PREDICTION_LIMIT = 1000 # if you want to just test the code, set to 5
debug_mode = False

if PREDICTION_LIMIT < 50:
    debug_mode = True

############################################################################

# Read the CSV file 
with open(input_csv_filename, newline='') as csvfile:
    data = list(csv.DictReader(csvfile))

# Define the system message
system_message = prep_msg

# Initialize a list to store the results
finetuned_predictions = []
true_values = []
countries = []

def query_model(row, window_start):
    country_name = row['country'] # loop through each country
    
    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years

    # regular GDP prompts
    input_message = f"The country of interest is: {country_name} and GDP (in Billions of US dollars at present day prices) from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    task_message = f". Predict the GDP (in Billions of US dollars at present day prices) for {country_name} in {prediction_year}: "

    # # normalized GDP prompts
    # input_message = f"We have normalized GDP values to be between 0 and 10. The country of interest is: {country_name} and normalized GDP from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    # task_message = f". Predict the normalized GDP for {country_name} in {prediction_year}: "

    # what the LLM should return
    true_value = float(row[str(i+predict_after_years)])

    # query the model
    completion = openai.ChatCompletion.create(
      model="ft:gpt-3.5-turbo-0613:personal::8Fmg2rfz",
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_message + task_message}
      ]
    )
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = completion.choices[0].message
    finetuned_prediction = float(finetuned_prediction_dict['content'])

    # preview results
    if debug_mode:
        print(input_message)
        print(task_message)
        print(finetuned_prediction, true_value)

    return country_name, finetuned_prediction, true_value

    # # Append the input and agent messages to the prompts list
    # return {
    #     "messages": [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": input_message + task_message},
    #         {"role": "assistant", "content": agent_message}
    #     ]
    # }

prediction_idx = 0
for row in data:
    if prediction_idx < PREDICTION_LIMIT:
        country_name, y_hat, y_true = query_model(row, test_windows_start) # Extract relevant data for that country and start year combination
        countries.append(country_name)
        true_values.append(y_true)
        finetuned_predictions.append(y_hat)
    prediction_idx = prediction_idx + 1

# store results in a dataframe and write to disk
results_df = pd.DataFrame()
results_df['country'] = countries
results_df['predicted_gdp'] = finetuned_predictions
results_df['observed_gdp'] = true_values
results_df.to_csv("../data/gdp/test_cleaned_gdp_finetuned_predictions.csv")