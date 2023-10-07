import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
df = pd.read_csv("C:/Users/navee/OneDrive/Desktop/car project/car project/car project/dataset.csv")
print(df)
print(df.describe())
print(df.info())

print(plt.hist(df['Year']))
print(plt.title("year plot"))
print(plt.xlabel("year"))
print(plt.ylabel("frequency"))
print(plt.show())

print(plt.hist(df['Present_Price']))
print(plt.title("price plot"))
print(plt.xlabel("present price"))
print(plt.ylabel("frequency"))
print(plt.show())

print(plt.hist(df['Kms_Driven']))
print(plt.title("Kms driven plot"))
print(plt.xlabel("Kms_Driven"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(x='Fuel_Type',data=df))
print(plt.title("Fuel type plot"))
print(plt.xlabel("Fuel type"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(x='Seller_Type',data=df))
print(plt.title("Seller type plot"))
print(plt.xlabel("Seller type"))
print(plt.ylabel("frequency"))
print(plt.show())

print(sns.countplot(x='Transmission',data=df))
print(plt.title("Transmission plot"))
print(plt.xlabel("Transmission"))
print(plt.ylabel("frequency"))
print(plt.show())


print(df.Fuel_Type.value_counts())
print(df.Seller_Type.value_counts())
print(df.Transmission.value_counts())

df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace({'Seller_Type':{'Dealer':0,'Individual':1,}},inplace=True)
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

print(df)


X=df.drop(['Car_Name','Selling_Price','Owner'],axis=1)
print(X)

y=df['Selling_Price']
print(y)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=2)

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

df1=pd.DataFrame({"Actual":y_test,"lr":y_pred})
print(df1)

accuracy=r2_score(y_test,y_pred)
print(accuracy)

print(plt.scatter(y_test,y_pred))
print(plt.xlabel("actual values"))
print(plt.ylabel("predicted values"))
print(plt.title("actual price vs prediction price"))
print(plt.show())

print(plt.plot(df1['Actual'].iloc[0:11],label='Actual'))
print(plt.plot(df1['lr'].iloc[0:11],label="Lr"))
print(plt.legend())
print(plt.show())


lr=LinearRegression()
lr.fit(X,y)
joblib.dump(lr,open("model.pkl",'wb'))
model=joblib.load(open("model.pkl",'rb'))

