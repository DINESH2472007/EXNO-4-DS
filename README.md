# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
<img width="479" height="527" alt="439173174-13d0f43f-b39d-4a97-bf03-68791a6196fe" src="https://github.com/user-attachments/assets/b5e7e987-52ed-4fae-aa93-68c59932056e" />

```python
df.head()
```
<img width="442" height="263" alt="439173374-8636587f-4c05-40a3-a7c9-24cc50682d35" src="https://github.com/user-attachments/assets/43a68b7b-9299-4070-9582-6d3c1c04e30f" />

```python
df.dropna()
```
<img width="519" height="522" alt="439173590-2535560d-bdae-4cc4-ba89-189e41efd284" src="https://github.com/user-attachments/assets/8d9f2de7-2364-4ad8-a5fb-4121dbf41235" />

```python
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

<img width="215" height="65" alt="439173831-1d320f4c-93a2-4eee-830b-0c1614785cd1" src="https://github.com/user-attachments/assets/990611b7-c177-41ad-96f7-9d09781946bb" />

```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="456" height="455" alt="439173990-ac4cf1e2-a97e-4175-95ec-b42ee138cdd6" src="https://github.com/user-attachments/assets/0f5122a8-7c40-47a5-86e3-9ca164521e99" />

```python
df1=pd.read_csv("bmi.csv")
df2=pd.read_csv("bmi.csv")
df3=pd.read_csv("bmi.csv")
df4=pd.read_csv("bmi.csv")
df5=pd.read_csv("bmi.csv")
df1
```
<img width="436" height="516" alt="439174570-1d4e78df-0e1d-4017-baee-cfdbb8abf647" src="https://github.com/user-attachments/assets/4260c7d9-6b45-4ba9-8157-42b745476cfa" />

```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
```
<img width="478" height="434" alt="439174723-6f278dd8-8da9-4187-b3b8-2073448ff1ad" src="https://github.com/user-attachments/assets/7901da5e-e216-43b2-9e1d-36bdcd70b549" />

```python
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
<img width="478" height="514" alt="439174913-6f99cd31-f513-4533-ad85-2b0b79ffe5c9" src="https://github.com/user-attachments/assets/489f68d8-7535-4c72-904c-4ffe02c57149" />

```python
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
```
<img width="444" height="521" alt="439175112-5d275238-b999-472c-bb67-0124a3520aa3" src="https://github.com/user-attachments/assets/615302e5-2c66-476f-90ae-d9cd9db95fd7" />

```python
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
df4
```
<img width="449" height="514" alt="439175349-97f2fafb-8f88-4ae5-8cbc-fa5991d6a575" src="https://github.com/user-attachments/assets/5cd1f125-c482-4c7d-beb7-0dad37c28a06" />

```python
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv("income(1) (1).csv")
data
```
<img width="1826" height="530" alt="439175614-6b9b38d5-d8f8-4dab-8fbc-d923385dc9ea" src="https://github.com/user-attachments/assets/aa5a74d4-0b80-4410-af9b-1b584c4500bb" />

```python
data1=pd.read_csv('/content/titanic_dataset (1).csv')
data1
```
<img width="1703" height="665" alt="436603679-40be0a5f-d7fc-46d1-8ce8-4e37a8bab56b" src="https://github.com/user-attachments/assets/14da7139-1961-4e98-9c28-a1cc6c7d5468" />

```python
data1=data1.dropna()
x=data1.drop(['Survived','Name','Ticket'],axis=1)
y=data1['Survived']
data1['Sex']=data1['Sex'].astype('category')
data1['Cabin']=data1['Cabin'].astype('category')
data1['Embarked']=data1['Embarked'].astype('category')
```

```python
data1['Sex']=data1['Sex'].cat.codes
data1['Cabin']=data1['Cabin'].cat.codes
data1['Embarked']=data1['Embarked'].cat.codes
```

```python
data1
```
<img width="1597" height="627" alt="436605776-03996577-41bb-4412-b31f-14a9e9c31ae7" src="https://github.com/user-attachments/assets/7bd65dcd-5d69-4cc7-bece-bd7b9a160994" />

```python
k=5
selector=SelectKBest(score_func=chi2,k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
```

```python
x_encoded=pd.get_dummies(x)
selector=SelectKBest(score_func=chi2,k=5)
x_new=selector.fit_transform(x_encoded,y)
```

```python
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="965" height="85" alt="436604781-d03e6b8e-25f9-4562-85b1-c32261ab1652" src="https://github.com/user-attachments/assets/bf3772b1-ff25-4897-9a4e-12a7171a8a3a" />

```python
selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1025" height="85" alt="436604940-42a89e33-d693-4945-b001-e472692a70ee" src="https://github.com/user-attachments/assets/4bb1ec8b-d041-418a-b15b-c8e0d75f549d" />


```python
selector=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1003" height="93" alt="436605188-a5ef2ad2-d54e-460f-956a-d95b9f9e1a7b" src="https://github.com/user-attachments/assets/73f73988-ce4c-4a18-b8d2-5e4e10357287" />

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
```

```python
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```

<img width="929" height="147" alt="436605344-d024fddb-3894-49c1-8a7b-c512d6f3a22a" src="https://github.com/user-attachments/assets/3236198b-eec1-415b-924a-1c333c3ebd95" />

```python
from sklearn.ensemble import RandomForestClassifier
```

```python
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_selection=model.feature_importances_
threshold=0.1
selected_features=x.columns[feature_selection>threshold]
print("Selected Features:")
print(selected_features)
```

<img width="961" height="101" alt="436605502-4c002a38-2b05-4764-bb85-e12acd17e8fc" src="https://github.com/user-attachments/assets/347b43bd-203e-4b07-aac8-53f49a10112e" />

```python
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance=model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance>threshold]
print("Selected Features:")
print(selected_features)
```

<img width="359" height="65" alt="436605573-115b5af5-5ec2-42b3-993f-e81e14e6b70d" src="https://github.com/user-attachments/assets/0cbbcef1-237b-43d1-b26e-74298acfce37" />

# RESULT:
Thus the feature selection and feature scaling has been used on the given dataset and executed successfully.

