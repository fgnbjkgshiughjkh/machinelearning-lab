import numpy as np
from collections import Counter
import math

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, binning_type=None, num_bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.binning_type = binning_type
        self.num_bins = num_bins
        self.tree = None

    def _entropy(self, y):
        """
        Calculate the entropy of a list of class labels.
        """
        counter = Counter(y)
        entropy = 0
        total = len(y)
        for label in counter:
            probability = counter[label] / total
            entropy -= probability * math.log2(probability)
        return entropy

    def _information_gain(self, X, y, feature_index):
        """
        Calculate the information gain for a specific feature.
        """
        # Calculate the entropy before splitting
        entropy_before_split = self._entropy(y)

        # Calculate the weighted average entropy after splitting
        counter = Counter(X[:, feature_index])
        total_samples = len(y)
        entropy_after_split = 0
        for value in counter:
            value_indices = np.where(X[:, feature_index] == value)[0]
            subset_entropy = self._entropy(y[value_indices])
            entropy_after_split += (len(value_indices) / total_samples) * subset_entropy

        # Calculate information gain
        information_gain = entropy_before_split - entropy_after_split
        return information_gain

    def _find_best_split(self, X, y):
        """
        Find the best feature to split on based on information gain.
        """
        best_feature_index = None
        best_information_gain = -1

        num_features = X.shape[1]
        for i in range(num_features):
            gain = self._information_gain(X, y, i)
            if gain > best_information_gain:
                best_information_gain = gain
                best_feature_index = i

        return best_feature_index

    def _split(self, X, y, feature_index, value):
        """
        Split the dataset based on a given feature and value.
        """
        mask = X[:, feature_index] == value
        return X[mask], y[mask]

    def _build_tree(self, X, y, depth):
        """
        Recursively build the Decision Tree.
        """
        if len(set(y)) == 1:
            return y[0]  # Leaf node, return the class label

        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]  # Leaf node, return the most common class label

        if len(X) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]  # Leaf node, return the most common class label

        best_feature_index = self._find_best_split(X, y)
        if best_feature_index is None:
            return Counter(y).most_common(1)[0][0]  # Leaf node, return the most common class label

        node = {}
        node['feature_index'] = best_feature_index
        node['children'] = {}

        unique_values = np.unique(X[:, best_feature_index])
        for value in unique_values:
            X_split, y_split = self._split(X, y, best_feature_index, value)
            node['children'][value] = self._build_tree(X_split, y_split, depth + 1)

        return node

    def fit(self, X, y):
        """
        Fit the Decision Tree to the training data.
        """
        if self.binning_type and self.num_bins:
            X = self._binning(X)

        self.tree = self._build_tree(X, y, depth=0)

    def _binning(self, X):
        """
        Perform binning on the input data based on the specified binning type.
        """
        if self.binning_type == 'equal_width':
            return self._equal_width_binning(X)
        elif self.binning_type == 'frequency':
            return self._frequency_binning(X)
        else:
            raise ValueError("Invalid binning type. Supported types are 'equal_width' and 'frequency'.")

    def _equal_width_binning(self, X):
        """
        Perform equal width binning on the input data.
        """
        for i in range(X.shape[1]):
            min_val = np.min(X[:, i])
            max_val = np.max(X[:, i])
            bin_width = (max_val - min_val) / self.num_bins
            bins = [min_val + j * bin_width for j in range(self.num_bins)]
            bins.append(max_val)
            X[:, i] = np.digitize(X[:, i], bins)
        return X

    def _frequency_binning(self, X):
        """
        Perform frequency binning on the input data.
        """
        for i in range(X.shape[1]):
            freqs, edges = np.histogram(X[:, i], bins=self.num_bins)
            X[:, i] = np.digitize(X[:, i], edges[:-1])
        return X

    def _predict_sample(self, x, tree):
        """
        Predict the class label for a single sample using the Decision Tree.
        """
        if isinstance(tree, dict):
            feature_index = tree['feature_index']
            value = x[feature_index]
            if value in tree['children']:
                child_tree = tree['children'][value]
                return self._predict_sample(x, child_tree)
            else:
                return None
        else:
            return tree

    def predict(self, X):
        """
        Predict the class labels for multiple samples using the Decision Tree.
        """
        if self.tree is None:
            raise ValueError("Decision Tree has not been trained. Fit the model first.")

        if self.binning_type and self.num_bins:
            X = self._binning(X)

        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x, self.tree))
        return predictions

# Example usage:
X_train = np.array([
    [5, 1],
    [6, 2],
    [7, 3],
    [8, 4],
    [9, 5],
    [10, 6],
    [11, 7],
    [12, 8]
])
y_train = np.array([0, 0, 1, 1, 0, 0, 1, 1])

X_test = np.array([
    [5, 1],
    [6, 2],
    [10, 6],
    [12, 8]
])

clf = DecisionTree(max_depth=3, min_samples_split=2, binning_type='equal_width', num_bins=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Predictions:", predictions)
