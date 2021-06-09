# from numpy.core.numeric import tensordot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data=pd.read_csv('Advertising.csv')
data=data.drop(['Unnamed: 0'],axis=1)
print(data.head())
print(data.corr())
print("from the above graph we can undertood that TV ads and sales have highest correlation")

sns.heatmap(data.corr(),annot=True)
plt.show()

x_train,x_test,y_train,y_test=train_test_split(data[['TV','radio','newspaper']],data['sales'],test_size=0.30,random_state=10)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


linear_regression =LinearRegression()
linear_regression.fit(x_train,y_train)
pred_sales=linear_regression.predict(x_test)

#creating a pickle file
pickle_out=open('salespred.pkl','wb')
pickle.dump(linear_regression,pickle_out)
pickle_out.close()