import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('iris.csv')
data.shape
data.head()
data.tail()
data.columns
data.info
data.describe()
sns.heatmap(data.loc[:,['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species']].corr(),annot=True)
plt.scatter(x=['SepalLengthCm'],y=['Species'],color='blue')
plt.scatter(x=['SepalWidthCm'],y=['Species'],color='yellow')
plt.scatter(x=['PetalLengthCm'],y=['Species'],color='black')
plt.scatter(x=['PetalWidthCm'],y=['Species'],color='green')


data['Species'].unique()
data['Species'].value_counts()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['Species']=encoder.fit_transform(data['Species'])


x=data.drop(['Species'],axis=1)
y=data['Species']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.coef_
regressor.intercept_


y_pred=regressor.predict(x_test)

from sklearn import metrics 
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.mean_absolute_error(y_test,y_pred)
metrics.r2_score(y_test,y_pred)
#####################################################
