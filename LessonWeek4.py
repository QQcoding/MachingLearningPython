import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('DataPreProcessing.csv')

# Picking only firt five rows for demonstration purpose
df_temp=df.head()

#print(df_temp.describe())
df_new=df_temp.drop(columns=['sex','age'])
#print(df_new)
df_new.loc[:,'BMI']=(df_new['Weight']/(df_new['Height']*df_new['Height']))*703
# print(df_new)
