

from sklearn import preprocessing 
lb = preprocessing.LabelBinarizer()
a=[0,0,0,0,0,0,1,1,1,1,1]
ab=lb.fit_transform(a)
print(ab)