from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris=load_iris()
print(type(iris))
labels=iris.target
print(labels)

#print(iris.target_names)
features=iris.data
labels=iris.target
#print (labels)
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2)
model =KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy {:.2f}".format(accuracy*100))