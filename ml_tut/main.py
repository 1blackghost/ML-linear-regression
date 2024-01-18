import tensorflow
import pandas as pd
import numpy as np
from sklearn  import linear_model
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data=pd.read_csv("student-mat.csv",sep=";")
data=data[["G1",'G2','G3','absences','failures']]

predict="G3"

X=np.array(data.drop([predict],axis=1))
Y=np.array(data[predict])
best=0

for _ in range(30):

	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

	linear=linear_model.LinearRegression()
	linear.fit(x_train,y_train)
	acc=linear.score(x_test,y_test)
	print(acc)
	if acc>best:
		best=acc
		with open("model.pickle","wb") as f:
			pickle.dump(linear,f)

print(best)
pickle_in=open("model.pickle","rb")
linear=pickle.load(pickle_in)


predictions=linear.predict(x_test)

for i in range(len(predictions)):
	print(predictions[i],x_test[i],y_test[i])