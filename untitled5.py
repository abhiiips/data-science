import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as mlt
import numpy as np  
import plotly.express as px
import plotly.io as pio
pio.renderers.default="browser"

data=pd.read_csv("Social_Network_Ads.csv")
data.shape
data.columns
data.drop(["User ID"],axis=1,inplace=True)
data.shape
data.describe()
################33

sns.boxplot(x=data["Gender"],y=data["Purchased"])
sns.countplot(x=data["Gender"],hue=data["Purchased"])

px.box(y=data["Age"],x=data["Purchased"])
sns.boxplot(x=data["Purchased"],y=data["Age"])

px.box(x=data["Purchased"],y=data["EstimatedSalary"])

px.histogram(data["EstimatedSalary"])

px.box(x=data["Purchased"],y=data["EstimatedSalary"],color=data["Gender"])

px.scatter_3d(x=data["Age"],y=data["EstimatedSalary"],z=data["Purchased"],color=data["Gender"])

#######3
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data["Gender"]=encoder.fit_transform(data["Gender"])
###split 
x=data.drop(["Purchased"],axis=1)
y=data["Purchased"]
############################################################################################################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)
################################################################################333
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
###################################################################################
y_pred=classifier.predict(x_test)
