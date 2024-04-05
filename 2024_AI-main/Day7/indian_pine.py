import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

uri = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

df = pd.read_csv(uri)

ddc = df.copy(deep=True)
ddc[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = ddc[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

print(ddc.isnull().sum())

ddc.dropna(inplace=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(ddc.copy().drop(["Outcome"], axis=1), ),
                        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                'BMI', 'DiabetesPedigreeFunction', 'Age'])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

cl = {}

cl["Knn1"] = KNeighborsClassifier(11)
cl["SVC"] = SVC()
cl["Knn2"] = KNeighborsClassifier(n_neighbors=8, metric='minkowski')
cl["tree"] = tree.DecisionTreeClassifier(criterion='gini')
cl["RForest"] = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=False)
cl["DeciTr"] = DecisionTreeClassifier()
cl["Logistic Regression"] = LogisticRegression()

y = ddc.loc[:, 'Outcome'].values
X = df_scaled.values

from sklearn.model_selection import train_test_split

X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=42)

pr = {}
for k, c in cl.items():
    c.fit(X_tr, y_tr)
    pr[k] = c.predict(X_t)

from sklearn.metrics import accuracy_score, classification_report

sc = {}
for k, c in cl.items():
    sc[k] = accuracy_score(y_t, pr[k])

for k, s in sc.items():
    print(f"Score= {s} for {k}")

for k, s in sc.items():
    print(f"Classification for: {k}")
    print(classification_report(y_t, pr[k]))
    
#------------------------------------------------------------------------------------------------------------------------------------------

# B Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train = torch.FloatTensor(X_tr)
x_test = torch.FloatTensor(X_t)
y_train = torch.LongTensor(y_tr)  
y_test = torch.LongTensor(y_t)    

class ANN_model(nn.Module):
    def __init__(self, input_features=8, hidden1=20, hidden2=10, hidden3=5, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, out_features)  
        
    def forward(self, x):
        x = F.relu(self.f_connected1(x))  
        x = F.relu(self.f_connected2(x)) 
        x = F.relu(self.f_connected3(x))
        x = self.out(x)
        return x

# To make the test repeatable
torch.manual_seed(42)

model = ANN_model()

loss_function = nn.CrossEntropyLoss()            
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 900
final_losses = []

# Train FIT
for i in range(epochs):
    y_pred = model.forward(x_train)  
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss.item())  
    if i % 1000 == 1:  
        print(f"Epoch: {i}, Loss: {loss.item()}")  
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
predictions = []
with torch.no_grad():  # Corrected no_grad()
    for data in x_test:  # Corrected x_test
        y_pred = model(data.unsqueeze(0))  # Corrected model input shape
        predictions.append(y_pred.argmax().item())  # Corrected append prediction

from sklearn.metrics import accuracy_score, classification_report

print("Report ANN:")
print("Accuracy:", accuracy_score(predictions, y_test))
print("Classification Report:")
print(classification_report(predictions, y_test))

        