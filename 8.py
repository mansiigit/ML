from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()

print("\nIRIS FEATURES \TARGET NAMES: \n", iris.target_names)
for i, name in enumerate(iris.target_names):
    print(f"\n[{i}]:[{name}]")

print("\nIRIS DATA :\n", iris.data)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

print("\nX TRAIN \n", X_train)
print("\nX TEST \n", X_test)
print("\nY TRAIN \n", y_train)
print("\nY TEST \n", y_test)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = kn.predict(x_new)
print("\nXNEW \n", x_new)
print(f"\nPredicted target value: {prediction}")
print(f"\nPredicted feature name: {iris.target_names[prediction][0]}")

for x, actual in zip(X_test, y_test):
    prediction = kn.predict([x])
    print(f"\nActual:[{actual}][{iris.target_names[actual]}], Predicted:{prediction[0]}[{iris.target_names[prediction][0]}]")

# Display test score
print(f"\nTEST SCORE[ACCURACY]: {kn.score(X_test, y_test):.2f}\n")
