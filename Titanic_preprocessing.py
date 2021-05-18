##################################################
# Titanic Dataset Preprocessing
##################################################

"""
Variable	Definition	              Key
survival	Survival	         0 = No, 1 = Yes
pclass	    Ticket class	     1 = 1st, 2 = 2nd, 3 = 3rd
sex	        Sex
Age	        Age in years
sibsp	    # of siblings / spouses aboard the Titanic
parch	    # of parents / children aboard the Titanic
ticket	    Ticket number
fare	    Passenger fare
cabin	    Cabin number
embarked    Port of Embarkation	    C = Cherbourg, Q = Queenstown, S = Southampton"""



import pandas as pd
import numpy as np
from dsmlbc4.helpers.helpers import *
from dsmlbc4.helpers.eda import *
from dsmlbc4.helpers.data_prep import *
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option("display.expand_frame_repr", False)

data = pd.read_csv("Week_7/titanic.csv")
df = data.copy()
df.head()


# EDA

check_df(df)
df.shape  # 891 observations, 12 variables

df.columns

# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')


# Numerical Variables
num_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]
# [ col for col in df.columns if df[col].dtype != "O"]

# Categorical Variables
cat_cols = [col for col in df.columns if df[col].dtype == "O"]

final_cat_cols, final_num_cols, num_but_cat, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# Observations: 891
# Variables: 12
# cat_cols: 6
# num_cols: 3
# cat_but_car: 3
# num_but_cat: 4


for col in final_cat_cols:
    cat_summary(df, col)





for col in final_num_cols:
    num_summary(df, col)



# --------------------
# FEATURE ENGINEERING
# --------------------

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'



df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int')
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([i for i in x.split() if i.startswith("Dr")]))
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 25), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 25) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 25), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 25) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'









#  AYKIRI GOZLEM

# check outlier
for col in num_cols:
    print(col, check_outlier(df, col))

# replace with thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

check_df(df)











#  LABEL ENCODING

binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

df.head()


#  ONE-HOT ENCODING

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


# STANDART SCALER


scaler = StandardScaler().fit(df[["AGE"]])
df["AGE"] = scaler.transform(df[["AGE"]])

check_df(df)