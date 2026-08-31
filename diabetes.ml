import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv")

dp = df.info()
dh = df.head()
dg = df.describe()
de = df.shape
dr = df.head(5)
dw = df.tail(5)
dl = df.isnull().sum()
dy = df.duplicated().sum()
dq = df[df.isnull().any(axis = 1)]

drop = df.drop_duplicates()
drop.to_csv("diabetes.csv" , index = False)
do = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

x = do.drop("diabetes" , axis =1)
y = do["diabetes"]
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.2 , random_state = 42 )
print("X_train : " , x_train.shape)
print("X_test : " , x_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


model = LogisticRegression()
model.fit(X_train_scaled , y_train )
y_train_pred = model.predict(X_train_scaled)
print("weights = " , model.coef_)
print("Bais = " , model.intercept_)
print("Accuracy = " , model.score(X_train_scaled , y_train))
print("Test Accuracy = ", model.score(X_test_scaled, y_test))


plt.scatter(X_train_scaled[y_train==0 , 0] , X_train_scaled[y_train == 0 , 4] , label = "Class 0" , color = "red")
plt.scatter(X_train_scaled[y_train==1 , 0] , X_train_scaled[y_train == 1 , 4] , label = "Class 1" , color = "blue")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend()
plt.title("Diabetes Classififcation")
plt.grid()
plt.show()



import joblib

joblib.dump(model, 'logistic.pkl')
joblib.dump(scaler, 'scaler.pkl')















