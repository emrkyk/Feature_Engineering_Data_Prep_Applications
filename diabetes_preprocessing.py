# =============================
# DIABETES DATA PREPROCESSING
# =============================

# Summary of Dataset
"""
In particular, all patients here are females at least 21 years old of Pima Indian heritage.

FEATURES:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history.
Age: Age
Outcome: 1 for Diabetes
"""

import pandas as pd
import numpy as np
from helpers.helpers import *
from helpers.eda import *
from helpers.data_prep import *
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option("display.expand_frame_repr", False)

data = pd.read_csv("Week_7/diabetes.csv")
df = data.copy()
df.head()

df.info()  # float64(2), int64(7)   # all variables => numerical
df.shape  # (768 row, 9 columns)

df.columns
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
#       dtype='object')

df.describe().T
#                            count    mean     std    min    25%     50%     75%     max
# Pregnancies              768.000   3.845   3.370  0.000  1.000   3.000   6.000  17.000
# Glucose                  768.000 120.895  31.973  0.000 99.000 117.000 140.250 199.000
# BloodPressure            768.000  69.105  19.356  0.000 62.000  72.000  80.000 122.000
# SkinThickness            768.000  20.536  15.952  0.000  0.000  23.000  32.000  99.000
# Insulin                  768.000  79.799 115.244  0.000  0.000  30.500 127.250 846.000
# BMI                      768.000  31.993   7.884  0.000 27.300  32.000  36.600  67.100
# DiabetesPedigreeFunction 768.000   0.472   0.331  0.078  0.244   0.372   0.626   2.420
# Age                      768.000  33.241  11.760 21.000 24.000  29.000  41.000  81.000
# Outcome                  768.000   0.349   0.477  0.000  0.000   0.000   1.000   1.000


df.isnull().values.any()  # False

check_df(df)  # no null-values , clean dataset

[num_summary(df, col, plot=True) for col in df.columns]
# application of num_summary function derived from eda.py module  (num_summmary func:  summary of numerical variables)

missing_values_table(df)

# --------------
# Missing Values
# --------------

df.head(50)
# existence of variables that have non-null values but some rows are equal to 0 which does not make sense.
for col in df.columns:
    print(col, df[df[col] == 0][col].count())

# Pregnancies 111
# Glucose 5
# BloodPressure 35
# SkinThickness 227
# Insulin 374
# BMI 11
# DiabetesPedigreeFunction 0
# Age 0
# Outcome 500     ====>  TARGET VARIABLE

# so let's create a list that represents these variables.
list_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in list_zeros:
    df[col].replace(0, np.NaN, inplace=True)  # np.NAN ?? sor
df.head(50)

df[list_zeros].hist(
    figsize=(20, 20))  # crucial to observe the distribution of variables with figures in feature engineering
plt.show()

# Which variables that have missing values should be filled with median or mean?
"""Normally distributed variables should be filled with mean, 
variables that may have outliers should be filled with medians."""

df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)

df.isnull().sum()

# ---------------------------
# Outliers / Aykırı Gözlemler
# ---------------------------
check_df(df)

for col in df.columns:
    print(col, check_outlier(df, col))

for col in df.columns:
    replace_with_thresholds(df, col)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
check_df(df)

# Creating new, meaningful variables, data points.

for col in ["BMI", "BloodPressure", "Glucose", "Age"]:
    print("variable:", col)
    print(df[col].describe().T)

df.loc[(df['BMI'] <= 18.5), 'NEW_BMI'] = 'Underweight'
df.loc[(df['BMI'] > 18) & (df['BMI'] <= 24.9), 'NEW_BMI'] = 'Normal weight'
df.loc[(df['BMI'] > 24.9) & (df['BMI'] <= 29.9), 'NEW_BMI'] = 'Overweight'
df.loc[(df['BMI'] > 29.9), 'NEW_BMI'] = 'Obesity'

df.loc[(df['BloodPressure'] < 80), 'NEW_BP'] = 'Normal'
df.loc[(df['BloodPressure'] >= 80) & (df['BloodPressure'] <= 89), 'NEW_BP'] = 'High Blood Pressure 1'
df.loc[(df['BloodPressure'] > 89), 'NEW_BP'] = 'High Blood Pressure 1'

df.loc[(df['Glucose'] <= 115), 'NEW_GLUCOSE'] = 'Excellent'
df.loc[(df['Glucose'] > 115) & (df['Glucose'] <= 180), 'NEW_GLUCOSE'] = 'Good'
df.loc[(df['Glucose'] > 180), 'NEW_GLUCOSE'] = 'Action Suggested'

df.loc[(df['Age'] <= 24), 'NEW_AGE'] = 'Young'
df.loc[(df['Age'] > 24) & (df['Age'] <= 40), 'NEW_AGE'] = 'Adult'
df.loc[(df['Age'] > 40) & (df['Age'] <= 60), 'NEW_AGE'] = 'Mature'
df.loc[(df['Age'] > 60), 'NEW_AGE'] = 'Senior'

check_df(df)

df.head()
df.columns

# --------------------------
# LABEL ENCODING                 # Conversions related to representation of variables
# --------------------------
# Expressing categorical variables ===> numerical
#  Neden? Bazı fonk. kategorik tipte değişkenler yerine bunları sayısal olarak temsil edebilecek bir versiyonunu ister.
# Özellikle 2 sınıflı değişkenleri labellarını değiştiriyoruz, binary encoding de denebilir.

binary_cols = [col for col in df.columns if
               len(df[col].unique()) == 2 and df[col].dtypes == 'O']  # kategorik ve 2 sınıflı değişkenleri seç
# ['NEW_BP']

df[df['NEW_BP'] == 'High Blood Pressure 1'].describe().T

for col in binary_cols:
    df = label_encoder(df, col)

check_df(df)
df.head()

# --------------------------
# ONE-HOT ENCODING
# --------------------------

# İkiden fazla sınıfa sahip olan kategorik değişkenlerin 1-0 olarak encode edilmesi.

onehot_e_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
# ['NEW_BMI', 'NEW_GLUCOSE', 'NEW_AGE']

df = one_hot_encoder(df, onehot_e_cols)

df.head()
df.shape
df.columns

# --------------------------
# STANDARD SCALER
# --------------------------
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s

# from sklearn.preprocessing import StandardScaler
#
# df['Pregnancies'] = df['Pregnancies'].astype(float)
#
#
# scaler = StandardScaler()
# #df = scaler.fit_transform(df["Pregnancies"])
#
# df = scaler.fit(df)
# df = scaler.transform(df)
#
# df.head()
# df.shape
#
# df.info()



# --------------------------
#  MINMAXSCALER
# --------------------------

from sklearn.preprocessing import MinMaxScaler

# num_cols_lim = [col for col in num_cols if df[col].nunique() > 20 and col not in "Salary"]

# scaler = MinMaxScaler(feature_range=(0, 1))
# df[num_cols_lim] = scaler.fit_transform(df[num_cols_lim])

scaler = MinMaxScaler()

# transform data

scaled = scaler.fit_transform(df)

df = pd.DataFrame(scaled, columns=df.columns)

df.head()
df.shape
