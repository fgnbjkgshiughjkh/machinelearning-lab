import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Step 1: Load the dataset
classes = pd.read_csv('C:\\Users\\harsh\\PycharmProjects\\ml\\venv\\Malayalam_Char_Gabor.csv')

# Step 2: Calculate mean and standard deviation for each feature
column_data = classes.iloc[:, 1:-1]
for column_name, class_data in column_data.items():
    mean = np.mean(class_data)
    print("Mean of feature", column_name, ":", mean)
    std = np.std(class_data)
    print("Standard deviation of feature", column_name, ":", std)

# Step 3: Calculate mean vectors for each class
class_means = {}
class_labels = classes.iloc[:, -1].unique()
for label in class_labels:
    class_means[label] = column_data[classes.iloc[:, -1] == label].mean()




# Step 4: Plot histogram for a selected feature and calculate mean and variance
feature_index = 0
feature_data = classes.iloc[:, feature_index]
plt.hist(feature_data, bins=10)
plt.title("Histogram of Feature " + str(feature_index))
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
mean = np.mean(feature_data)
variance = np.var(feature_data)
print("Mean of Feature", feature_index, ":", mean)
print("Variance of Feature", feature_index, ":", variance)

# Step 5: Split dataset into training and testing sets
X = classes.iloc[:, 1:-1]
y = classes.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Step 6: Train a kNN classifier (k = 3)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Step 7: Test the accuracy of the kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy of kNN classifier:", accuracy)

# Step 8: Use the predict() function to study the prediction behavior of the classifier for test vectors
predictions = neigh.predict(X_test)

# Step 9: Train a SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Step 10: Evaluate SVM classifier
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)

precision_train = precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
recall_train = recall_score(y_train, y_train_pred, average='weighted', zero_division=1)
f1_train = f1_score(y_train, y_train_pred, average='weighted', zero_division=1)

precision_test = precision_score(y_test, y_test_pred, average='weighted', zero_division=1)
recall_test = recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
f1_test = f1_score(y_test, y_test_pred, average='weighted', zero_division=1)

print("\nTraining set metrics (SVM):")
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1 Score:", f1_train)
print("\nTest set metrics (SVM):")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1 Score:", f1_test)

# Step 11: Make k = 1 to implement NN classifier and compare the results with kNN (k = 3). Vary k from 1 to 11 and make an accuracy plot.
accuracies = []
for k in range(1, 12):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    accuracies.append(accuracy)

plt.plot(range(1, 12), accuracies)
plt.title("Accuracy Plot for Different Values of k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()
