import pandas as pd
import csv
import json
import os
import time
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

###################### FILL IN THE RIGHT VALUES HERE  ######################

# input file - SPECIFY
prediction_mode_string = "_finetuned_"
ZERO_SHOT = True
if ZERO_SHOT:
    prediction_mode_string = "_zero_shot_"


# if working with GDP
input_csv_filename = "../data/gdp/test_cleaned_gdp.csv" #'../data/gdp/cleaned_gdp_df_countries.csv'
# output_csv_filename = " ../data/gdp/test_cleaned_gdp" + prediction_mode_string + "predictions.csv"
output_csv_filename = "test_cleaned_gdp" + prediction_mode_string + "predictions.csv"

# Train / test splits
test_windows_start = 2012
predict_after_years = 10


'''
# if working with GDP growth


# Train / test splits
prediction_horizon = 1 #1 #3
if prediction_horizon == 3:
    test_windows_start = 2010
    predict_after_years = 12
elif prediction_horizon == 1:
    test_windows_start = 2012 #2012 #2010
    predict_after_years = 10 #10 #12
else: 
    print("bad parameters")

input_csv_filename = "../data/gdp/rounded_cleaned_gdp_growth_df.csv" 
output_csv_filename = "../data/gdp/test_cleaned_gdp_growth" + prediction_mode_string + "predictions_" + str(prediction_horizon) + "yr.csv"
output_csv_filename = "test_cleaned_gdp_growth" + prediction_mode_string + "predictions_" + str(prediction_horizon) + "yr.csv"
'''
# system message to "prepare" the llm
nom_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years."
zero_shot_nom_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years. Respond using just a numeric prediction. Don't try to help me calculate this myself or give caveats, simply return a single number."
norm_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years. The GDP has been normalized so that it is always between 0 and 10."
zero_shot_nom_prep_msg = "You are a prediction agent predicting the GDP for a country in a future year when provided an input detailing the country's real GDP (in billions of present USD) for the last 10 years. Respond using just a numeric prediction. Don't try to help me calculate this myself or give caveats, simply return a single number."
growth_prep_msg = "You are a prediction agent predicting the GDP growth rates for a country in a future year when provided an input detailing the country's growth rate in real GDP for the last 10 years."
zero_shot_growth_prep_msg = "You are a prediction agent predicting the GDP growth rates for a country in a future year when provided an input detailing the country's growth rate in real GDP for the last 10 years. Respond using just a numeric prediction. Don't try to help me calculate this myself or give caveats, simply return a single number."
prep_msg = zero_shot_nom_prep_msg

# limit how many predictions to make
PREDICTION_LIMIT = 250 # if you want to just test the code, set to 5
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

'''
For predicting gdp values directly
'''

def query_model_gdp(row, window_start, prep_msg, zero_shot=False):
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
    model=""
    if zero_shot:
        model="gpt-3.5-turbo" #"gpt-3.5-turbo-1106" #"gpt-3.5-turbo"
    else:
        model="ft:gpt-3.5-turbo-0613:personal::8Fmg2rfz"

    # messages to user about what exactly is happening
    if debug_mode:
        print(input_message)
        print(task_message)
        print("Model being used for predictions is ", model)
        print("Predictions are being made for year ", prediction_year)
        print("First year for which data is being prompted in is ", window_start)
        
    # query the model
    completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_message + task_message}
            ]
    )
    
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = completion.choices[0].message

    # convert returned message to a float with some error handling 
    try:
        float_value_out = float(finetuned_prediction_dict['content'])
    except ValueError:
        print(f"Error: Invalid value encountered. Could not convert '{finetuned_prediction_dict['content']}' to a float.")
        float_value_out = None  
    finetuned_prediction = float_value_out

    # preview results
    print(country_name, finetuned_prediction, true_value)

    return country_name, finetuned_prediction, true_value, input_message, task_message

'''
For predicting GDP growth rates
'''

def query_model_gdp_growth(row, window_start, prep_msg=growth_prep_msg, zero_shot=False):
    '''
    to be used to query a model finetuned to make gdp growth 
    predictions in % terms on a 3 year horizon
    '''
    country_name = row['country'] # loop through each country
    model=""
    if zero_shot:
        model="gpt-3.5-turbo" #"gpt-3.5-turbo-1106"#"gpt-3.5-turbo"
    else:
        if prediction_horizon == 3:
            model="ft:gpt-3.5-turbo-1106:personal::8IU7aKVL"
        elif prediction_horizon == 1:
            model="ft:gpt-3.5-turbo-0613:personal::8KcBXc2t"
        else:
            print("Check parameters") 

    # Create sliding window prompts
    i = window_start
    prediction_year = i + predict_after_years

    # messages to user about what exactly is happening
    if debug_mode:
        print("Model being used for predictions is ", model)
        print("Predictions are being made for year ", prediction_year)
        print("First year for which data is being prompted in is ", window_start)

    # GDP growth prompts
    input_message = f"The country of interest is: {country_name} and growth rates in percent of real GDP seen in previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    task_message = f"Predict the GDP growth rate (in percent) for {country_name} in {prediction_year}: "

    # what the LLM should return
    true_value = float(row[str(i+predict_after_years)])

    # query the model
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_message + task_message}
      ]
    )
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = completion.choices[0].message
    
    # convert returned message to a float with some error handling 
    try:
        float_value_out = float(finetuned_prediction_dict['content'])
    except ValueError:
        print(f"Error: Invalid value encountered. Could not convert '{finetuned_prediction_dict['content']}' to a float.")
        float_value_out = None  
    finetuned_prediction = float_value_out

    # preview results
    print(country_name, finetuned_prediction, true_value)

    return country_name, finetuned_prediction, true_value, input_message, task_message

# loop through all countries and return results
prediction_idx = 0
print("Making predictions on dataset ", input_csv_filename)
print("Writing results to ", output_csv_filename)
print("Model is being used as agent for: ", prep_msg)

for row in data:
    if prediction_idx < PREDICTION_LIMIT:
        country_name, y_hat, y_true, input_message, task_message = query_model_gdp(row, test_windows_start, prep_msg=prep_msg, zero_shot=ZERO_SHOT) # Extract relevant data for that country and start year combination - GDP 
        # country_name, y_hat, y_true, input_message, task_message = query_model_gdp_growth(row, test_windows_start, prep_msg=prep_msg, zero_shot=ZERO_SHOT) # Extract relevant data for that country and start year combination - GDP Growth
        time.sleep(1)
        if prediction_idx == 0:
           print(input_message)
           print(task_message)
           print("First year for which data is being prompted in is ", test_windows_start)
           print("Zero shot", ZERO_SHOT)
        countries.append(country_name)
        true_values.append(y_true)
        finetuned_predictions.append(y_hat)
    prediction_idx = prediction_idx + 1

# store results in a dataframe and write to disk
results_df = pd.DataFrame()
results_df['country'] = countries
results_df['predicted_gdp'] = finetuned_predictions
results_df['observed_gdp'] = true_values
results_df.to_csv(output_csv_filename, index=False)