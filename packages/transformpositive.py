'''
Function to transform numbers in dataframe
to positive numbers

By Dr. Raymond Hoogendoorn
Copyright 2021
'''

def transformPositive(df):
    labels = df.columns.values
    for label in labels:
        try:
            df[label] = abs(df[label])
        except:
            df[label] = df[label]
    return df
