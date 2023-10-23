import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import shapiro
    import scipy.stats as stats
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('datasets/cookie_cats.csv')

df.head()

print(df.userid.nunique() == df.shape[0])

df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]].T

df.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"])

#oyunu hiç oynamayan 3994 kişi var
df.groupby("sum_gamerounds").userid.count().reset_index().head(20)


# How many users reached gate 30 & gate 40 levels?
df.groupby("sum_gamerounds").userid.count().loc[[30,40]]

# Looking at the summary statistics, the control and Test groups seem similar, but are the two groups
# statistically significant? We will investigate this statistically.

pd.DataFrame({"RET1_COUNT": df["retention_1"].value_counts(),
              "RET7_COUNT": df["retention_7"].value_counts(),
              "RET1_RATIO": df["retention_1"].value_counts() / len(df),
              "RET7_RATIO": df["retention_7"].value_counts() / len(df)})
#Elde tutma değişkenlerinin versiyona göre özet istatistiklerine bakıldığında
# ve sum_gamerounds ile karşılaştırıldığında gruplar arasında benzerlikler var.
# Ancak istatistiksel olarak anlamlı bir fark olup olmadığını görmek daha faydalı olacaktır.

df["Retention"] = np.where((df.retention_1 == True) & (df.retention_7 == True), 1,0)

df.groupby(["version", "Retention"])["sum_gamerounds"].agg(["count", "median", "mean", "std", "max"])
# iki grup arasında karşılaştırma yapıldığında, geri dönüşüm (retention) değişkenlerinin birleştirilmesi
# sonucunda özet istatistiklerin de benzer olduğunu belirtiyor. Başka bir deyişle, iki grup arasındaki
# geri dönüşüm oranlarına bakıldığında, bu oranların birbirine yakın veya aynı olduğu anlaşılıyor.

df["NewRetention"] = list(map(lambda x,y: str(x)+"-"+str(y), df.retention_1, df.retention_7))
df.groupby(["version", "NewRetention"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]).reset_index()


df.head()
# Define A/B groups
df["version"] = np.where(df.version == "gate_30", "A", "B")

groupA = df[df["version"] == "A"]
groupB = df[df["version"] == "B"]
groupB.head()

test_stat, pvalue = shapiro(df.loc[df["version"] == "A", "sum_gamerounds"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["version"] == "B", "sum_gamerounds"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#normal dagılmamıstır cunku p value = 0


test_stat, pvalue = mannwhitneyu(df.loc[df["version"] == "A", "sum_gamerounds"],
                                 df.loc[df["version"] == "B", "sum_gamerounds"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 1024331250.5000, p-value = 0.0502 so anlamlı bir farklılık yoktur.






df["Retention_1_score"] = np.where((df.retention_1 == True), 1,0)
df["Retention_7_score"] = np.where((df.retention_1 == True), 1,0)
df.head()
df.groupby("version").agg({"Retention_1_score": "sum", "Retention_7_score": "sum"})

test_stat, pvalue = shapiro(df.loc[df["version"] == "A", "Retention_1_score"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["version"] == "B", "Retention_1_score"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = mannwhitneyu(df.loc[df["version"] == "A", "Retention_1_score"],
                                 df.loc[df["version"] == "B", "Retention_1_score"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 1022682813.0000, p-value = 0.0744
#dolasıyla ıkı grup arasında 1 hafta sonra gırenler arasında istatisel bir fark var
#grup b daha avantajlı


test_stat, pvalue = shapiro(df.loc[df["version"] == "A", "Retention"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df.loc[df["version"] == "A", "Retention"],
                                 df.loc[df["version"] == "B", "Retention"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 1023112332.0000, p-value = 0.0072
#oyunu bitirmeleri anlamında, anlamlı bi farklılık vardır

df.head()



