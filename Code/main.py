import numpy as np
import matplotlib.pyplot as plt
import sys
import load_datasets
import NeuralNet  # importer la classe du Reseau de Neurones
import DecisionTree  # importer la classe de l'Arbre de Decision


# importer dautres fichiers et classes si vous en avez developpes
# importer dautres bibliotheques au besoin, sauf celles qui font du machine learning

decision_tree_iris = DecisionTree.DecisionTree()
decision_tree_congress = DecisionTree.DecisionTree()
decision_tree_monks1 = DecisionTree.DecisionTree()
decision_tree_monks2 = DecisionTree.DecisionTree()
decision_tree_monks3 = DecisionTree.DecisionTree()

# rn_iris = NeuralNet.NeuralNet()
# rn_congress = NeuralNet.NeuralNet()

# Charger/lire les datasets
(train_iris, train_labels_iris, test_iris, test_labels_iris) = load_datasets.load_iris_dataset(0.7)
(train_congress, train_labels_congress, test_congress, test_labels_congress) = load_datasets.load_congressional_dataset(0.7)
(train_monks1, train_labels_monks1, test_monks1, test_labels_monks1) = load_datasets.load_monks_dataset(1)
(train_monks2, train_labels_monks2, test_monks2, test_labels_monks2) = load_datasets.load_monks_dataset(2)
(train_monks3, train_labels_monks3, test_monks3, test_labels_monks3) = load_datasets.load_monks_dataset(3)

decision_tree_congress.learning_curve(train_congress, train_labels_congress, test_congress, test_labels_congress)

# # Entrainez votre classifieur
# decision_tree_iris.train(train_iris, train_labels_iris)
# decision_tree_congress.train(train_congress, train_labels_congress)
# decision_tree_monks1.train(train_monks1, train_labels_monks1)
# decision_tree_monks2.train(train_monks2, train_labels_monks2)
# decision_tree_monks3.train(train_monks3, train_labels_monks3)
# rn_iris.train(train_iris, train_labels_iris)
# rn_congress.train(train_congress, train_labels_congress)

# # Tester votre classifieur
# decision_tree_iris.test(test_iris, test_labels_iris)
# decision_tree_congress.test(train_congress, train_labels_congress)
# decision_tree_monks1.test(test_monks1, test_labels_monks1)
# decision_tree_monks2.test(test_monks2, test_labels_monks2)
# decision_tree_monks3.test(test_monks3, test_labels_monks3)
