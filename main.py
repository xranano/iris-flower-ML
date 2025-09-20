import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import os

iris_data = pd.read_csv('./Iris.csv')

def missing_values_describe(data):
    missing_value_stats = (data.isnull().sum()/len(data)*100)
    missing_value_col_count = sum(missing_value_stats>0)
    missing_value_stats = missing_value_stats.sort_values(ascending=False)[:missing_value_col_count]
    print("number of columns with missing values:", missing_value_col_count)
    if missing_value_col_count != 0:
        print("missing persentage:", missing_value_stats)
    else:
        print("No missing Data")

missing_values_describe(iris_data)

iris_data = iris_data.drop('Id', axis=1)

print("the dimension:", iris_data.shape)
print(iris_data.head())

print(iris_data.groupby('Species').size())


nameplot = iris_data['Species'].value_counts().plot.bar(title = "Flower class distribution", color = ['purple', 'pink', 'blue'])
nameplot.set_xlabel('class',size=20)
nameplot.set_ylabel('count',size=20)

iris_data.plot(kind='box', subplots=True,  layout=(2,2), sharex=False, sharey=False, title = 'box and whisker for each attribute')

iris_data.hist()
sns.set(style = 'ticks')
sns.pairplot(iris_data, hue = "Species")

x = iris_data.drop('Species', axis = 1)
y = iris_data['Species']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("Y_train.shape:", y_train.shape)
print("Y_test.shape:", y_test.shape)

plt.show()