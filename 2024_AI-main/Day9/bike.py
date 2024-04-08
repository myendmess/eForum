import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv("hour.csv")

# The `drop` method should take a list of columns to drop, and the `axis` parameter should be specified if more than one column is being dropped.
df1 = df.drop(["instant", "yr", "dteday"], axis=1)

# Instead of printing `df.info()` directly, assign it to a variable and then print it.
info = df.info()
print(info)

# Similarly, for `df1.describe()` and `df1.corr()`.
description = df1.describe()
correlation = df1.corr()
print(description)
print(correlation)

# For the `apply` method, you need to specify what operation you want to apply to each column.
unique_counts = df1.apply(lambda x: len(x.unique()))
print(unique_counts)

# The `isnull()` function needs to be chained with the `sum()` function to count the number of null values in each column.
null_check = df1.isnull().sum()
print(null_check)

X = df1.drop(["casual", "registered", "cnt", "atemp", "windspeed"], axis=1)
#print(X.corr())
y = df1["cnt"]
#print(X.head())

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=.20, random_state=42)

cl = {}
cl["lin"] = LinearRegression()
cl["Ridge"] = Ridge()
cl["Huber"] = HuberRegressor()
cl["Elan"] = ElasticNetCV()
cl["Dtre"] = DecisionTreeRegressor()
cl["forrest"] = RandomForestRegressor()
cl["GBoost"] = GradientBoostingRegressor()

for k, v in cl.items():
    v.fit(X_tr, y_tr)

y_p = {}
r2_ = {}
for k, v in cl.items():
    print(f"testing {k}")
    y_p[k] = v.predict(X_t)
    r2_[k] = r2_score(y_t, y_p[k])
    print(r2_[k])

    # To show data, uncomment the following lines
    # print(f"{k}: {r2_[k]} r2")

