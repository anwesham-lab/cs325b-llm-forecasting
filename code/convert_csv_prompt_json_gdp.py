import csv
import json

# Read the CSV file
with open('datasets/gdp/cleaned/cleaned_gdp_df_countries.csv', newline='') as csvfile:
    data = list(csv.DictReader(csvfile))

# Define the system message
system_message = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years."

# Initialize a list to store the prompts
prompts_json = []

###------
def generate_message(row):
    country_name = row['country']
    # Create sliding window prompts
    for i in [2001, 2006, 2011]:
        input_message = f"The country of interest is: {country_name} and GDP (in Billions of US dollars at present day prices) from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
        prediction_year = i + 10
        task_message = f". Predict the GDP (in Billions of US dollars at present day prices) for {country_name} in {prediction_year}: "
        prediction_amount = row[str(i+10)]
        agent_message = f"I predict the GDP for {country_name} in {prediction_year} will be {prediction_amount} (in Billions of US dollars at present day prices) "

        # Append the input and agent messages to the prompts list
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_message + task_message},
            {"role": "assistant", "content": agent_message}
        ]
    }    


# Loop through the data
for row in data:
    # Extract relevant data
    prompts_json.append(generate_message(row))
len_content = len(prompts_json)

# Save the prompts as a JSON file    
json_filename = "gdp.json"
with open(json_filename, 'w') as f:
    for entry in prompts_json:
        f.write(json.dumps(entry) + '\n')