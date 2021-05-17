
# =============================
# HITTERS DATA PREPROCESSING
# =============================

# Summary of Dataset
# AtBat: Number of times at bat in 1986
# Hits: Number of hits in 1986
# HmRun: Number of home runs in 1986
# Runs: Number of runs in 1986
# RBI: Number of runs batted in in 1986
# Walks: Number of walks in 1986
# Years: Number of years in the major leagues
# CAtBat: Number of times at bat during his career
# CHits: Number of hits during his career
# CHmRun: Number of home runs during his career
# CRuns: Number of runs during his career
# CRBI: Number of runs batted in during his career
# CWalks: Number of walks during his career
# League: A factor with levels A and N indicating player's league at the end of 1986
# Division: A factor with levels E and W indicating player's division at the end of 1986
# PutOuts: Number of put outs in 1986
# Assists: Number of assists in 1986
# Errors: Number of errors in 1986
# Salary: 1987 annual salary on opening day in thousands of dollars    ====>>> TARGET VARIABLE
# NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987

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
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option("display.expand_frame_repr", False)

data = pd.read_csv("Week_7/hitters.csv")
df = data.copy()
df.head()

df.shape  # (322 , 20)
df.describe().T

check_df(df)
df.info()

# Numerical Variables
num_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]

# Categorical Variables
cat_cols = [col for col in df.columns if df[col].dtype == "O"]

# -----------------
# # Missing Values - Eksik Gözlemler
# -----------------
missing_values_table(df)
#         n_miss  ratio
# Salary      59 18.320

df.info()
df.head(20)

df["Salary"].hist(figsize=(20, 20))
plt.show()

df["Salary"] = df["Salary"].fillna(df["Salary"].median())

# ---------------------------
# Outliers / Aykırı Gözlemler
# ---------------------------
df[num_cols].nunique()


num_cols_lim = [col for col in num_cols if df[col].nunique() > 20 and col not in "Salary"]

for col in num_cols_lim:
    print(col, check_outlier(df, col))

# AtBat False
# Hits False
# HmRun False
# Runs False
# RBI False
# Walks False
# Years False
# CAtBat False
# CHits False
# CHmRun False
# CRuns False
# CRBI False
# CWalks False
# PutOuts False
# Assists False
# Errors False


# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

check_df(df)
df.head()

df.loc[(df['Years'] < 5), 'Experience'] = 'inexperienced'
df.loc[(df['Years'] >= 5) & (df['Years'] < 10), 'Experience'] = 'experienced'
df.loc[(df['Years'] >= 10), 'Experience'] = 'senior'

df["Ratio_CAtBat"] = df["AtBat"] / df["CAtBat"]
df["Ratio_CHits"] = df["Hits"] / df["CHits"]
df["Ratio_CHmRun"] = df["HmRun"] / df["CHmRun"]
df["Ratio_Cruns"] = df["Runs"] / df["CRuns"]
df["Ratio_CRBI"] = df["RBI"] / df["CRBI"]
df["Ratio_CWalks"] = df["Walks"] / df["CWalks"]

df.shape  # 7 more variables added!

df.columns = [col.upper() for col in df.columns]
df.head()

# --------------------------
# LABEL ENCODING                 # Conversions related to representation of variables
# --------------------------
# Expressing categorical variables ===> numerical
#  Neden? Bazı fonk. kategorik tipte değişkenler yerine bunları sayısal olarak temsil edebilecek bir versiyonunu ister.
# Özellikle 2 sınıflı değişkenleri labellarını değiştiriyoruz, binary encoding de denebilir.

binary_cols = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique() == 2]

# ['LEAGUE', 'DIVISION', 'NEWLEAGUE']

for col in binary_cols:
    df = label_encoder(df, col)

check_df(df)

# --------------------------
# ONE-HOT ENCODING
# --------------------------

# İkiden fazla sınıfa sahip olan kategorik değişkenlerin 1-0 olarak encode edilmesi.
# Sadece 2 sınıfı olan değişkenlere de uygulanabilir.

onehot_e_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

df = one_hot_encoder(df, onehot_e_cols)

df.head()
df.shape
df.columns

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



