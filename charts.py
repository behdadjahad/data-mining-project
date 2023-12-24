import numpy as np
import pandas as pd
from statistics import stdev, mode, median
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


dataset_path = "./dataset/adult.csv"

df = pd.read_csv(dataset_path, encoding= 'unicode_escape', na_values='?')


#################################
# Create a count plot of income
sns.set(style="whitegrid")

sns.countplot(x="income", data=df, palette="Set2")
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()



##############################
# income sex count chart
grouped_data = df.groupby(['sex', 'income'])['income'].count().unstack()

# Create the stacked barplot
ax = grouped_data.plot(kind='bar', stacked=True, color=['#FFE1FF', '#BBFF96'])

# Add labels and title
ax.set_title('Income by Sex')
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.legend(['<=50K', '>50K'], loc='upper left')

plt.show()


#############################
# income race count chart
grouped_data = df.groupby(['race', 'income'])['income'].count().unstack()

# Create the stacked barplot
ax = grouped_data.plot(kind='bar', stacked=True, color=['#9C9CEE', '#8FB98F'])

# Add labels and title
ax.set_title('Income by Race')
ax.set_xlabel('Race')
ax.set_ylabel('Count')
ax.legend(['<=50K', '>50K'], loc='upper left')

plt.show()


############################
# income 3 col count chart
col=['workclass','education']
for i in col:
    k=pd.crosstab(df['income'],df[i])
    k.plot(kind='bar', stacked=True)
    plt.suptitle(i)
    plt.show()



#########################################################
# scatter plot for age and hours per week 
sns.scatterplot(x="age", y="education.num", hue="income", data=df, palette="Set2")
plt.title("Age vs. Education_num")
plt.xlabel("Age")
plt.ylabel("Education Num")
plt.show()


###################################################################
# ** find better distbution plot **
# distribution chart for attributes
for i in df.columns:
    sns.histplot(x=i, data=df, kde=False, bins=20) 
    plt.title(f"{i} Distribution")
    plt.xlabel(i)
    plt.ylabel("Count")
    plt.show()

df.hist(bins=20, figsize=(10, 8))
plt.show()


###################################################
# violin plot for sex and hourse per week attributes
sns.violinplot(x="income", y="education.num", hue="sex", data=df, split=True, inner="quart")
plt.title("education-num vs. Income by Sex")
plt.xlabel("Income")
plt.ylabel("education-num")
plt.show()


##################################################
# heatmap correlation matrix for numeric attributes
quantitative_variables = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(df[quantitative_variables]), columns=quantitative_variables)

print(data_normalized)
correlation_matrix = data_normalized.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Normalized Correlation Matrix of Quantitative Features')
plt.show()


###################################################
# pair plot (NOT RECOMMENDED)
sns.pairplot(df[['age', 'education.num', 'hours.per.week', 'income']], hue='income', palette="Set2")
plt.suptitle("Pairplot of Age, Education-num, and Hours-per-week by Income Level")

plt.show()



####################################################
# box plot for income age sex attributes (USE THE SCREENSHOT)
sns.boxplot(x="income", y="age", hue="sex", data=df, palette="Set3")
plt.title("Age Distribution by Income Level and Sex")
plt.xlabel("Income")
plt.ylabel("Age")
plt.show()

#########################################
# pie char for race attribute
plt.pie(df.race.value_counts(), labels=df.race.value_counts().index, autopct='%.0f%%', labeldistance=1.2, pctdistance=1.1);
plt.show()






#################################################
# pie chart for education attribute
counts=df['education'].value_counts().sort_index()
print(counts)
counts.plot(kind='pie',title='Education pie chart',figsize=(11,10))
plt.legend()
plt.show()





###################################################
#contingency
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