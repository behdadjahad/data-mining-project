import numpy as np
import pandas as pd
from statistics import stdev, mode, median
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


dataset_path = "./dataset/adult.csv"

df = pd.read_csv(dataset_path, encoding= 'unicode_escape', na_values='?')
# print(df.head())
# print(df.shape)
# print(df.info())

# print(df[["age", "income"]])

# print(df.columns)
# print(df["income"].value_counts())

# old = (df["age"] > 40)
# print(df.loc[old])
# print(df.describe())



############################################################################
# income sex count chart
grouped_data = df.groupby(['sex', 'income'])['income'].count().unstack()

# Create the stacked barplot
ax = grouped_data.plot(kind='bar', stacked=True, color=['#ae0100', '#00a500'])

# Add labels and title
ax.set_title('Income by Sex')
ax.set_xlabel('Sex')
ax.set_ylabel('Count')
ax.legend(['<=50K', '>50K'], loc='upper left')

plt.show()



########################################################
# income race count chart
grouped_data = df.groupby(['race', 'income'])['income'].count().unstack()

# Create the stacked barplot
ax = grouped_data.plot(kind='bar', stacked=True, color=['#FFC0CB', '#ADD8E6'])

# Add labels and title
ax.set_title('Income by Race')
ax.set_xlabel('Race')
ax.set_ylabel('Count')
ax.legend(['<=50K', '>50K'], loc='upper left')

plt.show()




################################################################
# income 3 col count chart
col=['workclass','education','sex']
for i in col:
    k=pd.crosstab(df['income'],df[i])
    k.plot(kind='bar', stacked=True)
    plt.show()



#####################################################################
# Create a count plot of income
sns.set(style="whitegrid")

sns.countplot(x="income", data=df, palette="Set2")
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()






#########################################################
# scatter plot for age and hours per week 
sns.scatterplot(x="age", y="hours.per.week", hue="income", data=df, palette="Set2")
plt.title("Age vs. Hours-per-Week")
plt.xlabel("Age")
plt.ylabel("Hours-per-Week")
plt.show()





###################################################################
# distribution chart for attributes
for i in df.columns:
    sns.histplot(x=i, data=df, kde=False, bins=20) 
    plt.title(f"{i} Distribution")
    plt.xlabel(i)
    plt.ylabel("Count")
    plt.show()

df.hist(bins=20, figsize=(10, 8))
plt.show()





#####################################################
# pie char for race attribute
plt.pie(df.race.value_counts(), labels=df.race.value_counts().index, autopct='%.0f%%', labeldistance=1.2, pctdistance=1.1);







##################################################
# pie chart for education attribute
counts=df['education'].value_counts().sort_index()
print(counts)
counts.plot(kind='pie',title='education_count',figsize=(11,10))
plt.legend()
plt.show()



###################################################
# violin plot for sex and hourse per week attributes
sns.violinplot(x="income", y="hours.per.week", hue="sex", data=df, split=True, inner="quart")
plt.title("Hours-per-Week vs. Income by Sex")
plt.xlabel("Income")
plt.ylabel("Hours-per-Week")
plt.show()


##################################################
# heatmap correlation matrix for numeric attributes
numeric_columns = df.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
corr = numeric_columns.corr()

# Create a heatmap of the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")

# Show the plot
plt.show()



###################################################
# pair plot
sns.pairplot(df[['age', 'education.num', 'hours.per.week', 'income']], hue='income', palette="Set2")
plt.suptitle("Pairplot of Age, Education-num, and Hours-per-week by Income Level")

plt.show()




####################################################
# box plot for income age sex attributes
sns.boxplot(x="income", y="age", hue="sex", data=df, palette="Set3")
plt.title("Age Distribution by Income Level and Sex")
plt.xlabel("Income")
plt.ylabel("Age")
plt.show()