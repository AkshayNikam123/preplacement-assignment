#!/usr/bin/env python
# coding: utf-8

# # Q-1. 
# Imagine you have a dataset where you have different Instagram features like username , Caption , Hashtag , Followers , Time_Since_posted , and likes , now your task is to predict the number of likes and Time Since posted and the rest of the features are your input features. Now you have to build a model which can predict the number of likes and Time Since posted. Dataset This is the Dataset You can use this dataset for this question.

# In[1]:


#Importing All important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Read the csv file 
df=pd.read_csv(r"C:\Users\akshay\Documents\instagram_reach.csv")


# In[2]:


df.head()


# In[3]:


#Dropping the Unnamed columns 
df.drop([ "Unnamed: 0","USERNAME"],axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


#Setting the S.No column as Indexing Column
df=df.set_index('S.No')


# In[7]:


df.columns


# In[9]:


df["Time since posted"] = df["Time since posted"].apply(lambda x:x.replace("hours","")).astype(float)


# In[10]:


# use label encoding on catigorical data
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()
cato = ["Caption","Hashtags"]
for i in cato:
    df[i] = lable.fit_transform(df[i])


# In[11]:


x= df.drop(["Time since posted","Likes"],axis=1)
y = df[["Time since posted","Likes"]]


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[14]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[15]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[16]:


def model_evaluation(test,pread):
    mse = mean_squared_error(test,pread)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test,pread)
    r2 = r2_score(test,pread)
    
    return mse,rmse,mae,r2


# In[17]:


forest = RandomForestRegressor()
forest.fit(X_train,y_train)


# In[18]:


forest.score(X_train,y_train)


# In[19]:


ypred = forest.predict(X_test)


# In[20]:


model_evaluation(y_test,ypred)


# # Q-2. 
# Imagine you have a dataset where you have different features like Age ,Gender , Height , Weight , BMI , and Blood Pressure and you have to classify the people into different classes like Normal , Overweight , Obesity , Underweight , and Extreme Obesity by using any 4 different classification algorithms. Now you have to build a model which can classify people into different classes. Dataset This is the Dataset You can use this dataset for this question.

# In[21]:


df1=pd.read_csv(r"C:\Users\akshay\Documents\ObesityDataSet_raw_and_data_sinthetic.csv")


# In[22]:


df1


# In[23]:


df1.isnull().sum()


# In[24]:


df1.duplicated().sum()


# In[25]:


df1.drop_duplicates(inplace=True)


# In[26]:


df1.info()


# In[27]:


# seprate numwerical and catigorical frature
catigorical_features = df1.select_dtypes(include="object").columns
numerical_features = df1.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[28]:


df1['NObeyesdad'].value_counts().plot.pie()


# In[29]:


# use label encoding on catigorical data
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()

for i in catigorical_features:
    df1[i] = lable.fit_transform(df1[i])


# In[30]:


plt.figure(figsize=(15,10))
sns.heatmap(df1.corr(),annot=True)


# In[31]:


x = df1.drop('NObeyesdad',axis=1)
y = df1['NObeyesdad']


# In[32]:


# saprate numwerical and catigorical frature
catigorical_features = x.select_dtypes(include="object").columns
numerical_features = x.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[33]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[34]:


## Numerical Pipline
num_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]
)

cato_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("scaler",StandardScaler())
    ]
)

# Create Preprocessor object
preprocessor = ColumnTransformer([
    ("num_pipline",num_pipline,numerical_features),
    ("cato_pipline",cato_pipline,catigorical_features)
])


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=25)


# In[36]:


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# In[37]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[38]:


# Logastic regression
logestic = LogisticRegression(class_weight="balanced")
logestic.fit(X_train,y_train)


# In[40]:


logestic.score(X_train,y_train)


# In[41]:


y_pred = logestic.predict(X_test)


# In[42]:


accuracy_score(y_test,y_pred)


# In[43]:


print(classification_report(y_test,y_pred))


# In[44]:


# Support Vector mechine
svm = SVC(C=10)
svm.fit(X_train,y_train)


# In[45]:


svm.score(X_train,y_train)


# In[46]:


y_pred = svm.predict(X_test)


# In[47]:


accuracy_score(y_test,y_pred)


# In[48]:


print(classification_report(y_test,y_pred))


# In[49]:


# Decision Tree
tree = DecisionTreeClassifier(max_depth=10,class_weight="balanced",min_samples_split=5)
tree.fit(X_train,y_train)


# In[50]:


tree.score(X_train,y_train)


# In[51]:


y_pred = tree.predict(X_test)


# In[52]:


accuracy_score(y_test,y_pred)


# In[53]:


print(classification_report(y_test,y_pred))


# In[54]:


# RandomForest
forest = RandomForestClassifier(n_estimators=200,max_depth=10,class_weight="balanced",min_samples_split=5)
forest.fit(X_train,y_train)


# In[55]:


forest.score(X_train,y_train)


# In[56]:


y_pred = forest.predict(X_test)


# In[57]:


accuracy_score(y_test,y_pred)


# In[58]:


print(classification_report(y_test,y_pred))


# # Q-3.
# Imagine you have a dataset where you have different categories of data, Now you need to find the most similar data to the given data by using any 4 different similarity algorithms. Now you have to build a model which can find the most similar data to the given data. Dataset This is the Dataset You can use this dataset for this question.

# In[60]:


import json
with open(r"C:\Users\akshay\Documents\News_Category_Dataset_v3.json",'r') as f:
    jdata = f.read()

jdata2  = [json.loads(line) for line in jdata.split('\n') if line]
data = pd.DataFrame.from_records(jdata2)


# In[61]:


data


# # Q-4.
# Imagine you working as a sale manager now you need to predict the Revenue
# and whether that particular revenue is on the weekend or not and find the
# Informational_Duration using the Ensemble learning algorithm
# Dataset This is the Dataset You can use this dataset for this question.

# In[62]:


data=pd.read_csv(r"C:\Users\akshay\Documents\online_shoppers_intention.csv")


# In[63]:


data


# In[64]:


data.isnull().sum()


# In[65]:


data.duplicated().sum()


# In[66]:


data.drop_duplicates(inplace=True)


# In[67]:


data.info()


# In[68]:


data["Revenue"] = data["Revenue"].map({True:1,False:0})
data['Weekend'] = data['Weekend'].map({True:1,False:0})


# In[69]:


# seperate numwerical and catigorical frature
catigorical_features = data.select_dtypes(include="object").columns
numerical_features = data.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[70]:


# use label encoding on catigorical data
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()

for i in catigorical_features:
    data[i] = lable.fit_transform(data[i])


# In[71]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True)


# In[72]:


x = data.drop("Revenue",axis=1)
y = data["Revenue"]


# In[73]:


# seperate numwerical and catigorical frature
catigorical_features = x.select_dtypes(include="object").columns
numerical_features = x.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[74]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[75]:


## Numerical Pipline
num_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]
)

cato_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("scaler",StandardScaler())
    ]
)

# Create Preprocessor object
preprocessor = ColumnTransformer([
    ("num_pipline",num_pipline,numerical_features),
    ("cato_pipline",cato_pipline,catigorical_features)
])


# In[76]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=25)


# In[77]:


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# In[78]:


# RandomForest
forest = RandomForestClassifier(n_estimators=250,max_depth=15,class_weight="balanced",min_samples_split=6)
forest.fit(X_train,y_train)


# In[79]:


forest.score(X_train,y_train)


# In[80]:


y_pred = forest.predict(X_test)


# In[81]:


accuracy_score(y_test,y_pred)


# In[82]:


print(classification_report(y_test,y_pred))


# In[83]:


from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(n_estimators=50)
bagging.fit(X_train,y_train)


# In[84]:


bagging.score(X_train,y_train)


# In[85]:


y_pred = bagging.predict(X_test)


# In[86]:


accuracy_score(y_test,y_pred)


# In[87]:


print(classification_report(y_test,y_pred))


# In[88]:


from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(n_estimators=50)
bagging.fit(X_train,y_train)


# In[89]:


bagging.score(X_train,y_train)


# In[90]:


y_pred = bagging.predict(X_test)


# In[91]:


accuracy_score(y_test,y_pred)


# In[92]:


print(classification_report(y_test,y_pred))


# # Q-5. 
# Uber is a taxi service provider as we know, we need to predict the high booking area using an Unsupervised algorithm and price for the location using a supervised algorithm and use some map function to display the data Dataset This is the Dataset You can use this dataset for this question.

# In[94]:


import pandas as pd
data=pd.read_csv(r"C:\Users\akshay\Documents\rideshare_kaggle.csv")


# In[95]:


sns.set(rc={"figure.figsize":(11,8)})
pd.pandas.set_option("display.max_columns",None)


# In[96]:


data.head()


# In[97]:


data.info()


# In[98]:


data.isnull().sum()


# In[99]:


data.duplicated().sum()


# In[100]:


data["price"] = data["price"].fillna(np.nanmedian(data["price"]))


# In[101]:


# separate numwerical and catigorical frature
catigorical_features = data.select_dtypes(include="object").columns
numerical_features = data.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[102]:


# use label encoding on catigorical data
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()

for i in catigorical_features:
    data[i] = lable.fit_transform(data[i])


# In[103]:


# Calculate the correlation matrix
correlation_matrix = data.corr().abs()

# Set the threshold for correlation value
threshold = 0.8

# Find highly correlated features
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] >= threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# Remove the correlated features from the DataFrame
data = data.drop(correlated_features, axis=1)

# Print the remaining features
print("Selected Features:")


# In[104]:


print(data.columns)


# In[105]:


x = data.drop("price",axis=1)
y = data["price"]


# In[106]:


#separate numerical and catigorical frature
catigorical_features = x.select_dtypes(include="object").columns
numerical_features = x.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[107]:


plt.figure(figsize=(50,30))
sns.heatmap(data.corr(),annot=True)


# In[108]:


## Numerical Pipline
num_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]
)

cato_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("scaler",StandardScaler())
    ]
)

# Create Preprocessor object
preprocessor = ColumnTransformer([
    ("num_pipline",num_pipline,numerical_features),
    ("cato_pipline",cato_pipline,catigorical_features)
])


# In[109]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=40)


# In[110]:


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# In[111]:


forest = RandomForestRegressor(n_estimators=300,max_depth=30,max_samples=10,max_leaf_nodes=8)
forest.fit(X_train,y_train)


# In[112]:


forest.score(X_train,y_train)


# In[113]:


y_pred = forest.predict(X_test)


# In[114]:


model_evaluation(y_test,y_pred)


# # Q-7. 
# Imagine you have a dataset where you need to predict the Genres of Music using an Unsupervised algorithm and you need to find the accuracy of the model, built-in docker, and use some library to display that in frontend Dataset This is the Dataset You can use this dataset for this question.

# In[115]:


import pandas as pd
data=pd.read_csv(r"C:\Users\akshay\Documents\data.csv")


# In[116]:


data.head()


# In[117]:


data.isnull().sum()


# In[118]:


data.drop(columns='filename',axis=1,inplace=True)


# In[119]:


data.duplicated().sum()


# In[120]:


data.drop_duplicates


# In[121]:


data["label"].value_counts()


# In[122]:


# separate numwerical and catigorical frature
catigorical_features = data.select_dtypes(include="object").columns
numerical_features = data.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[123]:


# use label encoding on catigorical data
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()

for i in catigorical_features:
    data[i] = lable.fit_transform(data[i])


# In[124]:


plt.figure(figsize=(30,20))
sns.heatmap(data.corr(),annot=True)


# In[125]:


x = data.drop('label',axis=1)
y = data['label']


# In[126]:


# separate numwerical and catigorical frature
catigorical_features = x.select_dtypes(include="object").columns
numerical_features = x.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[127]:


## Numerical Pipline
num_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]
)

cato_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("scaler",StandardScaler())
    ]
)

# Create Preprocessor object
preprocessor = ColumnTransformer([
    ("num_pipline",num_pipline,numerical_features),
    ("cato_pipline",cato_pipline,catigorical_features)
])


# In[128]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# In[129]:


X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# In[130]:


# logestic Regression
logestic = LogisticRegression(class_weight="balanced",C=10)
logestic.fit(X_train,y_train)


# In[131]:


logestic.score(X_train,y_train)


# In[132]:


y_pred = logestic.predict(X_test)


# In[133]:


accuracy_score(y_test,y_pred)


# In[134]:


from sklearn.ensemble import GradientBoostingClassifier
# gredient boosting
gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)


# In[135]:


gb.score(X_train,y_train)


# In[136]:


y_pred = gb.predict(X_test)


# In[137]:


accuracy_score(y_test,y_pred)


# # Q-6. 
# Imagine you have a dataset where you have predicted loan Eligibility using any
# 4 different classification algorithms. Now you have to build a model which can
# predict loan Eligibility and you need to find the accuracy of the model and built-in
# docker and use some library to display that in frontend
# Dataset This is the Dataset You can use this dataset for this question.

# In[138]:


data=pd.read_csv(r"C:\Users\akshay\Downloads\train_u6lujuX_CVtuZ9i (1).csv")


# In[139]:


data.head()


# In[140]:


data.isnull().sum()


# In[141]:


import statistics as st
data["Gender"] = data["Gender"].fillna(st.mode(data["Gender"]))
data["Married"] = data["Married"].fillna(st.mode(data["Married"]))
data["Self_Employed"] = data["Self_Employed"].fillna(st.mode(data["Self_Employed"]))


# In[142]:


data["LoanAmount"] = data["LoanAmount"].fillna(np.nanmedian(data["LoanAmount"]))
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(np.nanmedian(data["Loan_Amount_Term"]))
data["Credit_History"] = data["Credit_History"].fillna(np.nanmedian(data["Credit_History"]))


# In[143]:


data.drop(["Loan_ID","Dependents"],axis=1,inplace=True)


# In[145]:


# separate numwerical and catigorical frature
catigorical_features = data.select_dtypes(include="object").columns
numerical_features = data.select_dtypes(exclude="object").columns
print(catigorical_features)
print(numerical_features)


# In[146]:


# use label encoding on catigorical data
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()

for i in catigorical_features:
    data[i] = lable.fit_transform(data[i])


# In[147]:


data


# In[148]:


sns.heatmap(data.corr(),annot=True)


# In[149]:


x = data.drop("Loan_Status",axis=1)
y = data["Loan_Status"]


# In[150]:


## Numerical Pipline
num_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]
)

cato_pipline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("scaler",StandardScaler())
    ]
)

# Create Preprocessor object
preprocessor = ColumnTransformer([
    ("num_pipline",num_pipline,numerical_features),
    ("cato_pipline",cato_pipline,catigorical_features)
])


# In[151]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# In[152]:


# logestic Regression
logestic = LogisticRegression(class_weight="balanced",C=10)
logestic.fit(X_train,y_train)


# In[153]:


logestic.score(X_train,y_train)


# In[154]:


y_pred = logestic.predict(X_test)


# In[155]:


accuracy_score(y_test,y_pred)


# In[156]:


print(classification_report(y_test,y_pred))


# In[157]:


# Bagging using tree
tree = DecisionTreeClassifier()
bagging = BaggingClassifier(tree,n_estimators=50,max_samples=8)
bagging.fit(X_train,y_train)


# In[158]:


bagging.score(X_train,y_train)


# In[159]:


y_pred = bagging.predict(X_test)


# In[160]:


#Adaboost
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=150,learning_rate=0.1)
adaboost.fit(X_train,y_train)


# In[161]:


adaboost.score(X_train,y_train)


# In[162]:


y_pred = adaboost.predict(X_test)


# In[163]:


accuracy_score(y_test,y_pred)


# In[164]:


print(classification_report(y_test,y_pred))


# In[165]:


#Randomforest
forest = RandomForestClassifier(n_estimators=120,max_depth=8,class_weight="balanced",max_samples=2,max_leaf_nodes=3)
forest.fit(X_train,y_train)


# In[166]:


forest.score(X_train,y_train)


# In[167]:


y_pred = forest.predict(X_test)


# In[168]:


accuracy_score(y_test,y_pred)


# In[169]:


print(classification_report(y_test,y_pred))


# In[ ]:




