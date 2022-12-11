import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from tabulate import tabulate

data = pd.read_csv('iris_data.csv')

data_y = data['species']
data_x = data.drop(columns = ['species'])

x_train,x_test,y_train,y_test= train_test_split(data_x,data_y,test_size=0.4,random_state=0)

results_table = [['Method','Mean Score','Standard Deviation']]

model = svm.SVC(kernel='linear', C=1)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
results_table.append(['40% holdout',score,'N/A'])

model = svm.SVC(kernel='linear',C=1,random_state=42)
score = cross_val_score(model,data_x,data_y,cv=5)
results_table.append(['5 fold CV',np.mean(score),np.std(score)])

n_samples = data_x.shape[0]
cv = ShuffleSplit(n_splits=5,test_size=0.3,random_state=0)
score = cross_val_score(model,data_x,data_y,cv=cv)
results_table.append(['5 fold CV\n(passing cross validation iterator)',np.mean(score),np.std(score)])

print(tabulate(results_table,headers='firstrow',tablefmt='fancy_grid'))
