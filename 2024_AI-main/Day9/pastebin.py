import numpy as np
import pandas as pd

df = pd.read_csv("Fake data.txt")

print(df.head())
print(df.info())
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting histograms for each numerical column
numerical_columns = df.select_dtypes(include=np.number).columns
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Plotting pairplot to visualize relationships between numerical variables
sns.pairplot(df)
plt.show()

# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


X = df.iloc[:,:-1] # until mark
Y = df.iloc[:,-1:] #markl


from sklearn.model_selection import train_test_split
X_tr,X_t,Y_tr,Y_t = train_test_split(X,Y, test_size=20, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_tr,Y_tr)
y_predict = model.predict(X_t)
my_tst = model.predict([[6,2]])

print(f"Predict [[6,2]] = {my_tst}")

from sklearn.metrics import r2_score
r2 = r2_score(y_predict,Y_t)
print(f"R2_score: {r2}")