import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filepath_or_buffer = url, header = None, sep = ',', names = names)

array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1, shuffle = True)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pickle.dump(classifier, open('./Desplegar/LRClassifier.pkl', 'wb'))
