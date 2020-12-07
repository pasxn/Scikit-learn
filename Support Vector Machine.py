from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

classes = ['Setosa', 'Versicolor', 'Virginica']
iris = datasets.load_iris()
# split in features and labels
x = iris.data
y = iris.target

# train - test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#model
model = svm.SVC()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

# printing the output
print(predictions)
print(accuracy)

