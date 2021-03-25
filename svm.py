import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Brest Cancer Dataset.csv')
X = dataset.iloc[:,6:12].values
y = dataset.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.25, random_state = 0)

from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
cm = confusion_matrix(y_test, y_pred)
pr = precision_score(y_test, y_pred,average='weighted')
kt = recall_score(y_test, y_pred,average='weighted')
gt = f1_score(y_test, y_pred,average='weighted')

print(cm)
print(pr)
print(kt)
print(gt)




