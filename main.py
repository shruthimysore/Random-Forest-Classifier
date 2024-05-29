import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import numpy as np
from statistics import mode


class TreeNode:
    def __init__(self):
        self.type = None
        self.attribute = None
        self.classID = None
        self.children = []

    def set_leaf(self, classID):
        self.type = 'leaf'
        self.attribute = None
        self.classID = classID
        self.children = []

    def set_decision_node(self, attribute):
        self.type = 'decision'
        self.attribute = attribute
        self.classID = None
        self.children = []

class DecisionTree:
    def __init__(self, data, tree_depth=5):
        self.root = None
        self.attributeValueList = dict()
        self.tree_depth = tree_depth
        self.current_depth = 0

    def build_values_for_attribute(self, data, attribute_list):
        for attribute in attribute_list:
            values = list(set(data[attribute]))
            self.attributeValueList[attribute] = sorted(values)
        return

    def build_tree(self, data, attribute_list, min_instances=5):
        # Base conditions
        node = TreeNode()
        if (len(set(data['target'])) <= 1) or len(data) < min_instances:
            node.set_leaf(data.iloc[0]['target'])
            return node
        if (self.current_depth == self.tree_depth):
            #node.set_leaf(data.target.mode()[0])
            node.set_leaf(mode(data.target))
            return node
        self.current_depth += 1
        best_attribute,info_gain = self.find_best_split(data, attribute_list)
        # print("Best attribute: " +best_attribute+"    Info gain: " + str(info_gain))
        node.set_decision_node(best_attribute)
        #attribute_list.remove(best_attribute)

        for value in self.attributeValueList[best_attribute]:
            split_data = data[data[best_attribute] == value]
            if len(split_data) == 0:
                #node.set_leaf(data.target.mode()[0])
                node.set_leaf(mode(data.target))
                return node
            else:
                sub_tree = self.build_tree(split_data, attribute_list, min_instances)
                node.children.append(sub_tree)
        return node

    def entropy_of_split(self, data):
        base = 2.0
        entropy = 0.0
        data = list(data['target'])
        if len(data) == 0:
            return 0.0
        size = len(data)
        values, counts = [], []
        for d in data:
            if d not in values:
                values.append(d)
                counts.append(data.count(d))
        probability = [c / size for c in counts]
        probability = [p for p in probability if p != 0]
        entropy = -(sum([p * (math.log(p, base)) for p in probability]))
        return entropy

    def find_best_split(self, data, attributes):
        entropy_of_full_set = self.entropy_of_split(data)
        best_attribute = ''
        max_information_gain = 0
        selected_attr_count = int(math.sqrt(len(attributes)))
        selected_attributes = random.sample(attributes, selected_attr_count)

        for attribute in selected_attributes:
            unique_attribute_values = list(set(data[attribute]))
            count_of_attribute_values = [0] * len(unique_attribute_values)
            for i in range(len(unique_attribute_values)):
                count_of_attribute_values[i] = len(data[data[attribute] == unique_attribute_values[i]])
            entropy_for_attribute = 0.0
            for i in range(len(unique_attribute_values)):
                entropy_of_attribute_value = self.entropy_of_split(data[data[attribute] == unique_attribute_values[i]])
                entropy_for_attribute = entropy_for_attribute + (count_of_attribute_values[i] * entropy_of_attribute_value)
            entropy_for_attribute = entropy_for_attribute / sum(count_of_attribute_values)
            information_gain = entropy_of_full_set - entropy_for_attribute

            if (information_gain > max_information_gain):
                max_information_gain = information_gain
                best_attribute = attribute
        if max_information_gain == 0:
            return attributes[0], max_information_gain
        return best_attribute, max_information_gain


    def predict_class(self, current_node, datapoint):
        if current_node.type == 'leaf':
            return current_node.classID
        this_attribute = current_node.attribute
        return self.predict_class(current_node.children[datapoint[this_attribute]], datapoint)


    def test_predictions(self, X):
        X = pd.DataFrame(X)
        predictions = [False] * len(X)
        # correct_predictions = 0
        for i in range(len(X)):
            predictions[i] = self.predict_class(self.root,X.iloc[i])
            # if predictions[i]:
            #     correct_predictions += 1
        return predictions


class RandomForest:
    def __init__(self, num_of_trees, X_train, X_test, Y_train, Y_test):
        self.num_trees = num_of_trees
        self.dataset = dataset
        self.trees = []
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def fit(self, X_train, Y_train):
        # Bootstrap sample from the training data
        for i in range(self.num_trees):
            X_bootstrap, Y_bootstrap = self.bootstrap_data(X_train, Y_train)
            # Train a decision tree on the bootstrap sample
            bootstrap_data = X_bootstrap.join(Y_bootstrap)
            decision_tree = DecisionTree(bootstrap_data, tree_depth=5)
            attributeList = list(X_bootstrap.columns)
            decision_tree.build_values_for_attribute(bootstrap_data, attributeList)
            decision_tree.root = decision_tree.build_tree(bootstrap_data, attributeList)
            self.trees.append(decision_tree)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            test_predictions = tree.test_predictions(X)
            predictions.append(tuple(test_predictions))
        final_predictions = mode(predictions)
        return final_predictions


    def compute_accuracy(self, predictions, Y):
        correct_predictions = 0
        total_predictions = len(Y)
        for i in range(total_predictions):
            if Y[i] == predictions[i]:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return accuracy


    def bootstrap_data(self, X_train, y_train):
        n_samples = X_train.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_X = pd.DataFrame(X_train[indices])
        bootstrap_y = pd.DataFrame(y_train[indices], columns=['target'])
        return bootstrap_X, bootstrap_y

    def evaluate(self, y_true, y_pred):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if y_true[i] == 1:
                    false_negatives += 1
                else:
                    false_positives += 1
        accuracy = (true_positives + true_negatives) / len(y_true)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return accuracy, precision, recall, f1


def cross_validation(X, y, num_trees, k):
    # Split data into k folds
    n_samples = len(y)
    fold_size = n_samples // k
    fold_indices = np.arange(n_samples)
    np.random.shuffle(fold_indices)
    fold_indices = np.array_split(fold_indices, k)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for i in range(k):
        # Split data into training and testing sets
        test_indices = fold_indices[i]
        train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])
        X_train, Y_train = X.iloc[train_indices].values, y.iloc[train_indices].values
        X_test, Y_test = X.iloc[test_indices].values, y.iloc[test_indices].values
        # Train model on training set
        model = RandomForest(num_trees, X_train, X_test, Y_train, Y_test)
        model.fit(X_train, Y_train)
        # Evaluate model on testing set
        y_pred = model.predict(X_test)
        accuracy, precision, recall, f1 = model.evaluate(Y_test, y_pred)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(precision)
        f1_scores.append(f1)
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1_score = np.mean(f1_scores)
    return mean_accuracy, mean_precision, mean_recall, mean_f1_score


dataset = pd.read_csv('hw3_house_votes_84.csv')
dataset = pd.concat([dataset.iloc[:, 1:], dataset.iloc[:, 0]], axis=1)
dataset = dataset.rename(columns={'class': 'target'})
dataWithoutTarget = dataset.drop(labels='target',axis=1,inplace=False)
k = 10
num_trees = [1, 5, 10, 20, 30, 40, 50]
accuracy_plot, precision_plot, recall_plot, f1_plot = [], [], [], []
for tree in num_trees:
    accuracy, precision, recall, f1 = cross_validation(dataWithoutTarget, dataset.target, tree, k)
    accuracy_plot.append(accuracy)
    precision_plot.append(precision)
    recall_plot.append(recall)
    f1_plot.append(f1)

print("Accuracy: ",accuracy_plot)
print("Precision: ",precision_plot)
print("Recall: ",recall_plot)
print("F1: ",f1_plot)

plt.plot(num_trees, accuracy_plot)
plt.title('Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()

plt.plot(num_trees, precision_plot)
plt.title('Precision vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Precision')
plt.show()

plt.plot(num_trees, recall_plot)
plt.title('Recall vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Recall')
plt.show()

plt.plot(num_trees, f1_plot)
plt.title('F1 vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('F1')
plt.show()

# print("Accuracy: "+str(accuracy))
# print("Precision: "+str(precision))
# print("Recall: "+str(recall))
# print("F1 score: "+str(f1))




