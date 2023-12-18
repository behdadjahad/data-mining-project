import numpy as np
import pandas as pd
from statistics import stdev, mode, median
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


dataset_path = "./dataset/adult.csv"

df = pd.read_csv(dataset_path, encoding= 'unicode_escape', na_values='?')
print(df.head())
print(df.shape)
central_tendency = dict()

central_tendency["age"] = dict()
central_tendency["age"]["maen"] = df["age"].mean()
central_tendency["age"]["min"] = df["age"].min()
central_tendency["age"]["max"] = df["age"].max()
central_tendency["age"]["mode"] = mode(df["age"])
central_tendency["age"]["median"] = median(df["age"])
central_tendency["age"]["variance"] = df["age"].var()
central_tendency["age"]["standard_deviation"] = stdev(df["age"])


central_tendency["hours.per.week"] = dict()
central_tendency["hours.per.week"]["mean"] = df["hours.per.week"].mean()
central_tendency["hours.per.week"]["min"] = df["hours.per.week"].min()
central_tendency["hours.per.week"]["max"] = df["hours.per.week"].max()
central_tendency["hours.per.week"]["mode"] = mode(df["hours.per.week"])
central_tendency["hours.per.week"]["median"] = median(df["hours.per.week"])
central_tendency["hours.per.week"]["variance"] = df["hours.per.week"].var()
central_tendency["hours.per.week"]["standard_deviation"] = stdev(df["hours.per.week"])


central_tendency["capital.gain"] = dict()
central_tendency["capital.gain"]["mean"] = df["capital.gain"].mean()
central_tendency["capital.gain"]["min"] = df["capital.gain"].min()
central_tendency["capital.gain"]["max"] = df["capital.gain"].max()
central_tendency["capital.gain"]["mode"] = mode(df["capital.gain"])
central_tendency["capital.gain"]["median"] = median(df["capital.gain"])
central_tendency["capital.gain"]["variance"] = df["capital.gain"].var()
central_tendency["capital.gain"]["standard_deviation"] = stdev(df["capital.gain"])


central_tendency["capital.loss"] = dict()
central_tendency["capital.loss"]["mean"] = df["capital.loss"].mean()
central_tendency["capital.loss"]["min"] = df["capital.loss"].min()
central_tendency["capital.loss"]["max"] = df["capital.loss"].max()
central_tendency["capital.loss"]["mode"] = mode(df["capital.loss"])
central_tendency["capital.loss"]["median"] = median(df["capital.loss"])
central_tendency["capital.loss"]["variance"] = df["capital.loss"].var()
central_tendency["capital.loss"]["standard_deviation"] = stdev(df["capital.loss"])


central_tendency["education.num"] = dict()
central_tendency["education.num"]["mean"] = df["education.num"].mean()
central_tendency["education.num"]["min"] = df["education.num"].min()
central_tendency["education.num"]["max"] = df["education.num"].max()
central_tendency["education.num"]["mode"] = mode(df["education.num"])
central_tendency["education.num"]["median"] = median(df["education.num"])
central_tendency["education.num"]["variance"] = df["education.num"].var()
central_tendency["education.num"]["standard_deviation"] = stdev(df["education.num"])



for k in central_tendency:
    plt.figure(figsize=(12, 6))
    sns.histplot(df[k], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {k}')
    plt.xlabel(k)
    plt.ylabel('Frequency')
    plt.show()
    print(f"{k}:")
    for ct in central_tendency[k]:
        print(f"\t{ct}: {central_tendency[k][ct]}")
    print()


quantitative_variables = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(df[quantitative_variables]), columns=quantitative_variables)

print(data_normalized)
correlation_matrix = data_normalized.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Normalized Correlation Matrix of Quantitative Features')
plt.show()

categorical_variable = 'education'

frequency_table = df[categorical_variable].value_counts()
relative_frequency_table = df[categorical_variable].value_counts(normalize=True)

plt.figure(figsize=(12, 6))
sns.countplot(x=categorical_variable, data=df, order=frequency_table.index)
plt.title(f'Frequency of {categorical_variable}')
plt.xlabel(categorical_variable)
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

print(f'Frequency Table for {categorical_variable}:\n{frequency_table}\n')
print(f'Relative Frequency Table for {categorical_variable}:\n{relative_frequency_table}\n')



categorical_variable1 = 'education'
categorical_variable2 = 'occupation'

contingency_table = pd.crosstab(df[categorical_variable1], df[categorical_variable2])


observed_chi2 = contingency_table.values
n = observed_chi2.sum()
expected_chi2 = np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) / n
chi2 = ((observed_chi2 - expected_chi2) ** 2 / expected_chi2).sum()
phi2 = chi2 / n
r, k = contingency_table.shape
phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
rcorr = r - ((r - 1) ** 2) / (n - 1)
kcorr = k - ((k - 1) ** 2) / (n - 1)
cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt='d')
plt.title(f'Contingency Table of {categorical_variable1} and {categorical_variable2}')
plt.xlabel(categorical_variable2)
plt.ylabel(categorical_variable1)
plt.show()


print(f'Cramér\'s V: {cramers_v:.4f}')