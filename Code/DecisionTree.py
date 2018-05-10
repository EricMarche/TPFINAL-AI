"""
Vous allez definir une classe pour chaque algorithme que vous allez developper,
votre classe doit contenit au moins les 3 methodes definies ici bas,
    * train     : pour entrainer le modele sur l'ensemble d'entrainement
    * predict     : pour predire la classe d'un exemple donne
    * test         : pour tester sur l'ensemble de test
vous pouvez rajouter d'autres methodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les methodes train, predict et test de votre code.
"""

import numpy as np
from matplotlib import pyplot as plt
#Import pour calculer le temps d execution de test
import timeit
from collections import defaultdict, Counter

class DecisionTree:

    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre parametres au besoin,
        c'est a vous d'utiliser vos propres notations
        """
        self.dataset = []
        self.tree = None
    def train(self, train, train_labels): #vous pouvez rajouter d'autres attribus au besoin
        """
        c'est la methode qui va entrainer votre modele,
        train est une matrice de taille nxm, avec
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caracteristiques)

        train_labels : est une matrice de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        ------------

        """
        self.dataset = np.concatenate((train, train_labels), axis=1)

        #Notre liste d'attribut sera l'index des attributs
        attrs = []
        for i in range(0, len(train[0])):
            attrs.append(i)

        self.tree = self.decision_tree(self.dataset, attrs)

    def predict(self, exemple, label):
        """
        Predire la classe d'un exemple donne en entree
        exemple est de taille 1xm

        si la valeur retournee est la meme que la veleur dans label
        alors l'exemple est bien classifie, si non c'est une missclassification

        """
        prediction = None
        node = self.tree
        while prediction is None:
            if node.is_leaf():
                prediction = node.label
            else:
                value = exemple[node.attr]
                if value in node:
                    node = node[value]
                #Sa se peut quil est une erreur et on peut pas determiner la prediction
                #Donc on va mettre la prediction a -1 ou 0 je sais pas
                else :
                    prediction = 0.
        return prediction

    def test(self, test, test_labels):
        """
        c'est la methode qui va tester votre modele sur les donnees de test
        l'argument test est une matrice de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caracteristiques)

        test_labels : est une matrice taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les donnees de test, et afficher :
        - la matrice de confision (confusion matrix)
        - l'accuracy (ou le taux d'erreur)

        Bien entendu ces tests doivent etre faits sur les donnees de test seulement

        """
        predictions = []
        start = timeit.default_timer()
        for i in range(0, len(test)):
            prediction = self.predict(test[i], test_labels[i])
            predictions.append(prediction)
        confusion_matrix(predictions, test_labels)
        stop = timeit.default_timer()
        print "execution time : ",stop - start


    def decision_tree(self, data, attrs):
        if not attrs:
            result = same_class(data)
            root_node = Leaf(result)
        else:
            classes_dict = label_count(data)
            classes = classes_dict.keys()
            if len(classes_dict) == 1:
                #Nous retourne une liste des classes mais vu qu'il y a seulement
                #Une valeur possible on prend la premiere
                root_node = Leaf(classes[0])
            else:
                best_attr = self.best_attribute(data, attrs, classes)
                root_node = Node(best_attr)
                best_attr_values = {row[best_attr] for row in data}
                for value in best_attr_values:
                    rows = [row for row in data if row[best_attr] == value]
                    child = self.decision_tree(rows, attrs)
                    root_node[value] = child
        return root_node

    def best_attribute(self, data, attrs, classes):
        gain_factors = [(self.information_gain(data, attr, classes), attr)
                        for attr in attrs]
        gain_factors.sort()
        best_attribute = gain_factors[-1][1]
        attrs.pop(attrs.index(best_attribute))
        return best_attribute

    def information_gain(self, data, attr, classes):
        nb_data = len(data)
        partition = defaultdict(list)
        for row in data:
            partition[row[attr]].append(row)
        attr_entropy = 0.0
        for partition in partition.values():
            #fait une liste avec les classes
            sub_classes = [s[-1] for s in partition]
            attr_entropy += (len(partition) / nb_data) * entropy(sub_classes)

        return entropy(classes) - attr_entropy

    def learning_curve(self, train, train_labels, test, test_labels):
        k = 10
        n = len(train)
        x = []
        y = []
        predictions = []
        for i in range(1, k):
            print "test : ", self.tree
            # Si on veut un des split_test qui change
            # split_train, split_test = fold_split(train, i * (n / k), (i + 1) * (n / k))
            # split_train_labels, split_test_labels = fold_split(train_labels, i * (n / k), (i + 1) * (n / k))

            split_train, split_test = fold_split(train, (n / k), (i + 1) * (n / k))
            split_train_labels, split_test_labels = fold_split(train_labels, (n / k), (i + 1) * (n / k))
            print "split_train len : ", len(split_train), " split_test len  : ", len(split_test)

            self.train(split_train, split_train_labels)
            for i in range(0, len(split_train)):
                prediction = self.predict(split_train[i], split_train_labels[i])
                predictions.append(prediction)
            accuracy = total_accuracy(predictions, split_train_labels)
            x.append(len(split_train))
            y.append(accuracy)
            self.tree = None
            self.dataset = []

        plt.plot(x, y)
        plt.xlabel('Dataset size')
        plt.ylabel('Accuracy')
        plt.title('Model response to dataset size')
        plt.show()

def confusion_matrix(predictions, test_labels):
    labels = set(test_label[0] for test_label in test_labels)

    predictions = np.array(predictions).astype(int)
    test_list = np.hstack(test_labels).astype(int)
    matrix = np.zeros((len(labels), len(labels)))
    for a, p in zip(test_list, predictions):
        matrix[a][p] += 1

    false_positive = matrix.sum(axis=0) - np.diag(matrix)
    false_negative = matrix.sum(axis=1) - np.diag(matrix)
    true_positive = np.diag(matrix)
    true_negative = matrix.sum() - (false_negative + false_positive + true_positive)

    accuracy = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)
    print "accuracy: ", accuracy

    print "confusion matrix : "
    print matrix

    return matrix

def total_accuracy(predictions, test_labels):
    labels = set(test_label[0] for test_label in test_labels)

    predictions = np.array(predictions).astype(int)
    test_list = np.hstack(test_labels).astype(int)
    matrix = np.zeros((len(labels), len(labels)))
    for a, p in zip(test_list, predictions):
        matrix[a][p] += 1

    accuracy = sum(np.diag(matrix)) / float(len(test_labels))
    print "accuracy: ", accuracy
    return accuracy


def same_class(data):
    classes = [row[-1] for row in data]
    counter = Counter(classes)
    k, = counter.most_common(n=1)
    return k[0]

def entropy(data):
    nb_data = float(len(data))
    counter = Counter(data)
    sum_entropy = sum(-1.0*(counter[c] / nb_data)*np.log2(counter[c] / nb_data) for c in counter)
    return sum_entropy

def label_count(data):
    counts = {}
    for row in data:
        label = row[-1]
        if label is not counts.keys():
            counts[label] = 1
        counts[label] += 1
    return counts

def fold_split(data, start, end):
    start = int(start)
    end = int(end)

    train = data[start:end]
    test = data[:start]
    return train, test

class Node(dict):
    def __init__(self, attr, *args, **kwargs):
        self.attr = attr
        super(Node, self).__init__(*args, **kwargs)

    def is_leaf(self):
        return False

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.attr)

class Leaf(dict):
    def __init__(self, label, *args, **kwargs):
        self.label = label
        super(Leaf, self).__init__(*args, **kwargs)

    def is_leaf(self):
            return True

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.label)
