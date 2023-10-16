import numpy as np
import pandas as pd
from numpy import random


def load_csv():
    df1 = pd.read_excel("Predictions.xlsx")
    df1.head()

    df2 = pd.read_excel("oasis_longitudinal_demographics.xlsx")
    df2.head()

    return df1, df2


def find_unknowns(df):
    df = df.replace('?', np.NaN)
    print('Number of missing values:')
    for col in df.columns:
        print('\t%s: %d' % (col, df[col].isna().sum()))
    return df


def replace_with_removal(df):
    print('Number of rows in original data = %d' % (df.shape[0]))
    df = df.dropna()
    print('Number of rows after discarding missing values = %d' % (df.shape[0]))
    return df

def replace_with_mean(df):
    for i in df:
        if i == "SES" or i == "MMSE":
            mean_value = df[i].mean()
            print("Replacing NaN values of", i, "with mean of:", mean_value)
            df[i].fillna(value=mean_value, inplace=True)
    return df


def drop_unneeded(df):
    for i in df:
        if i == "Hand":
            print("Dropping column:", i)
            df = df.drop(["Hand"], axis=1)
    return df


def drop_dups(df):
    print('Number of rows in original data = %d' % (df.shape[0]))
    df = df.drop_duplicates()
    print('Number of rows after discarding duplicated values = %d' % (df.shape[0]))
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictions, demographics = load_csv()
    predictions = find_unknowns(predictions)
    demographics = find_unknowns(demographics)
    demographics = replace_with_mean(demographics)
    predictions = drop_unneeded(predictions)
    demographics = drop_unneeded(demographics)
    predictions = drop_dups(predictions)
    demographics = drop_dups(demographics)

    print()
    print("Predicitons: ")
    print(predictions)
    print()
    print("oasis_longitudinal_demographics: ")
    print(demographics)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
