#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:30:03 2023

@author: Nunoo Emmanuel Felix Landlord
"""

import pandas as pd

# import dataset
file_path = 'dataFS.csv'  
dff = pd.read_csv(file_path)

# data cleaning

#remove ID column
df_c = dff.drop(columns=['ID'])

#remove _(2) in povline
df_c["povline"] = df_c["povline"].str.replace("_(2)", "", regex=False)

# check for duplicates
total_duplicates = df_c.duplicated().sum()
print("Total Duplicates:", total_duplicates)

# remove all duplicates
#df_deduplicated = df.drop_duplicates()

# check for missing values
total_missing = df_c.isnull().sum().sum()
print("Total Missing Values:", total_missing)

# remove all rows with missing values
df = df_c.dropna()

#save new csv
df.to_csv('FS_cleaned.csv',index = False)
