import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('datasets/AB_Test_Data.csv')
df.head()

df.USER_ID.value_counts()

df.drop_duplicates(inplace=True)

for ver in df["VARIANT_NAME"].unique():
    s,p = shapiro(df.loc[df['VARIANT_NAME']==ver,"REVENUE"])
    print(f"Variant: {ver} \nStatistic: {s:.3f}\np-Value: {p:.3f}\n")

"""Variant: variant 
Statistic: 0.033
p-Value: 0.000
Variant: control 
Statistic: 0.022
p-Value: 0.000
"""
#normallik kosulu saglanmadıgı için menwithyu kullanılır

s,p = mannwhitneyu(df.loc[df['VARIANT_NAME'] == "variant","REVENUE"],
                    df.loc[df['VARIANT_NAME'] == "control","REVENUE"])
print(f"\nStatistic: {s:.3f}\np-Value: {p:.3f}\n")


"""Statistic: 7850692.000
p-Value: 0.513
"""

#anlamlı bir farklılık yoktur