import pandas as pd
from darts.datasets import ILINetDataset
import os.path

def create_ILI_data_csvs():
    if not os.path.isfile("../data/ILI/ILI_Data.csv"):
        series = ILINetDataset().load()
        series.to_csv("../data/ILI/ILI_Data.csv")
    df = pd.read_csv('../data/ILI/ILI_Data.csv')
    filtered_columns = df.filter(regex='^DATE|^AGE|^ILITOTAL', axis=1)
    df = df[filtered_columns.columns]
    df.reset_index(drop=True, inplace=True)
    new_df = df.T
    new_df.columns = new_df.iloc[0]
    new_df = new_df[1:]
    df.reset_index(drop=True, inplace=True)
    new_df = new_df.loc[:, ~new_df.columns.str.startswith('1997')]
    new_df.to_csv('../data/ILI/transposed_ILI_Data.csv')

def clean_ILI_data():
    df = pd.read_csv('../data/ILI/transposed_ILI_Data.csv')
    df.rename(columns={ df.columns[0]: "Demographic" }, inplace=True)
    dg_col = df['Demographic']
    df_vals = df.iloc[:, 1:]
    df_vals.columns = pd.to_datetime(df_vals.columns, format='%Y-%m-%d')
    summed_df = df_vals.groupby([df_vals.columns.year, df_vals.columns.month], axis=1).sum()
    new_columns = ['Demographic'] + [(str(year) + "-" + str(month)) for (year, month) in summed_df.columns]
    result_df = pd.DataFrame(columns=new_columns)
    result_df['Demographic'] = dg_col
    result_df.iloc[:, 1:] = summed_df.values
    result_df.to_csv('../data/ILI/cleaned_monthly_ILI.csv')
    # dg_col = new_df['Demographic']
    # df_vals = new_df.iloc[:, 1:]
    # df_vals.columns = pd.to_datetime(df_vals.columns, format='%Y-%m-%d')
    # new_df['Quarter'] = new_df.iloc[:, 1:].columns.to_series().dt.to_period("Q").values
    # summed_df = df_vals.groupby(df_vals.columns.year, axis=1).sum()
    # new_columns = ['Demographic'] + [str(year) for year in summed_df.columns]
    # result_df = pd.DataFrame(columns=new_columns)
    # result_df['Demographic']=dg_col
    # result_df.iloc[:, 1:] = summed_df.values
    # result_df.to_csv('../data/ILI/cleaned_ILI.csv')

if __name__=="__main__": 
    # create_ILI_data_csvs() 
    clean_ILI_data()