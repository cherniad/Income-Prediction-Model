import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Loading the file
df = pd.read_csv('/Users/Dascherry/Desktop/income.csv')

# Exploring the data

# Types of education
df.education.value_counts() 

# Types of workclass
df.workclass.value_counts()

# Types of occupation
df.occupation.value_counts()

# Transforming long data into wide data 

# Turning the occupation values into binary features
pd.get_dummies(df.occupation) 

# Renaming the column 
pd.get_dummies(df.occupation).add_prefix('occupation_')

# Dropping the occupation column and creating new column for each occupation with binary values 
df = pd.concat([df.drop('occupation', axis=1),pd.get_dummies(df.occupation).add_prefix('occupation_')], axis=1)

# Repeating the process for the workclass, marital status, relationship, race, native country columns
df = pd.concat([df.drop('workclass', axis=1),pd.get_dummies(df.workclass).add_prefix('workclass_')], axis=1)
df = pd.concat([df.drop('marital-status', axis=1),pd.get_dummies(df['marital-status']).add_prefix('marital-status_')], axis=1)
df = pd.concat([df.drop('relationship', axis=1),pd.get_dummies(df.relationship).add_prefix('relationship_')], axis=1)
df = pd.concat([df.drop('race', axis=1),pd.get_dummies(df.race).add_prefix('race_')], axis=1)
df = pd.concat([df.drop('native-country', axis=1),pd.get_dummies(df['native-country']).add_prefix('native-country_')], axis=1)

# Education column doesn't have values to be transformed into the binary format, hence dropping the column
df = df.drop('education', axis=1)
# fnlwgt column is how many people belong to the group, so we'll drop it
df = df.drop('fnlwgt', axis=1)
df 

# Now there are 48842 rows × 92 columns instead of 48842 rows × 15 columns in the beginning 

# Transforming the values in gender and income columns into binary features 
df['gender'] = df['gender'].apply(lambda x: 1 if x=='Male' else 0)
df['income'] = df['income'].apply(lambda x: 1 if x=='>50K' else 0)

# Checking the data 
df.columns.values

df['income']

# Creating a heatmap with maplotlib and seaborn
plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')

# There are too many features, so we will filter them out

# Calculating the absolute correlation coefficient between each column in df and the income column
correlations = df.corr()['income'].abs()

# Sorting the values in ascending order
sorted_correlations = correlations.sort_values()

# Taking 80% of the total number of columns and converting them to an integer
num_cols_to_drop = int(0.8 * len(df.columns))

# Selecting the indexes of the columns with the weakest correlations (80%) 
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index

# Dropping the columns with the weakest correlations
df_dropped = df.drop(cols_to_drop, axis=1)
df_dropped #48842 rows × 19 columns

# Creating new heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df_dropped.corr(), annot=True, cmap='vlag')

# Some observations: 
# 1. There is a positive correlation between the income and marital status
# 2. There is a positive correlation between the income and relationship_Husband
# 3. There is a positive correlation between the income and education, as well as age 
# 	 (whether you're educated or not and older, the income is higher)
# 4. There is a positive correlation between the income and gender 
#	 (being a man seems beneficial as man is 1 in df)

# There are a lot of binary features that are very similar to the decision tree
# so we'll use the random forest


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 20% of the data is going to be used
train_df, test_df = train_test_split(df, test_size=0.2)

train_df

test_df

# Splitting into x and y data

train_x = train_df.drop('income', axis=1)
train_y = train_df['income']

test_x = test_df.drop('income', axis=1)
test_y = test_df['income']

# Fitting the model
forest = RandomForestClassifier()
forest.fit(train_x, train_y)

# Model score
forest.score(test_x, test_y)

# Importance score for each feature in the df
forest.feature_importances_

# Zipping the scores with their corresponding names in the dictionary
importances = dict(zip(forest.feature_names_in_, forest.feature_importances_))
importances = {k: v for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}
importances

# The older you are and the more you work, the more you earn

# Using hyper parameter tuning to see which combination of parameteres gives the best performance

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 250],
    'max_depth':  [5, 10, 30, None],
    'min_samples_split': [2,4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                          param_grid=param_grid, verbose=10)

model = grid_search.fit(train_x, train_y)

# Best combination of parameteres 
grid_search.best_params_

# The best estimator
grid_search.best_estimator_

# Creating a visualisation with top-10 most important features in relation to income

importances_series = pd.Series(importances) # Using the dictionary created before

# Df with feature names and importances

feature_importance_df = pd.DataFrame({'Feature': train_x.columns, 'Importance': importances_series})

# Sorting top-10 of the most important features

top_10 = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
top_10_final = top_10.sort_values(by='Importance',ascending=True)


# Bar chart
importance = top_10_final['Importance']
feature = top_10_final['Feature']
fig = plt.figure(figsize=(10,6))
plt.barh(feature, importance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.show()

# Age, education and gender seem to be the most important features

# Showing partial dependence of the three most important features and 
# how the income changes in accordance with the variety in those three features
from sklearn.inspection import plot_partial_dependence
feature_names = ['age', 'educational-num', 'gender']
plot_partial_dependence(model, train_x, features=feature_names)


# Individual plot
from sklearn.inspection import plot_partial_dependence
feature_names = ['age']
plot_partial_dependence(model, train_x, features=feature_names)



