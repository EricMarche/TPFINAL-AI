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

class DecisionTree:

    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre parametres au besoin,
        c'est a vous d'utiliser vos propres notations
        """
        self.labels = []

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
        self.labels = self.label_counts(train_labels)
        examples = self.create_examples(train, train_labels)

        #On a un liste des attributs utilise.
        attrs = []
        for i in range(0, len(train)):
            attrs.append(i)

        print "examples : ", examples

    def predict(self, exemple, label):
        """
        Predire la classe d'un exemple donne en entree
        exemple est de taille 1xm

        si la valeur retournee est la meme que la veleur dans label
        alors l'exemple est bien classifie, si non c'est une missclassification

        """

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




    def decision_tree(examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return parent_examples
        elif len(examples) == 1:
            return
        elif all(attr == False  for attr in attrs):
            return
        else:
            best = choose_attribute(attrs, examples)
            tree = DecisionFork(best, dataset.attrnames[best], plurality_value(examples))
            for (v_k, exs) in best:
                subtree = decision_tree(
                    exs, removeall(best, attrs), examples)
                tree.add(v_k, subtree)
            return tree

    def create_examples(self, train, labels):
        examples = []
        for i in range (0, len(set(labels))):
            examples.append([])
        for i in range(0, len(labels)):
            examples[labels[i]].append(train[i])
        return examples


    #On va faire une liste selon les attributs
    def split_by_attr(self, train):
        train_by_attr = []
        nb_attr = train[0]
        #on prend le nombre d'attribut dune donnees
        for i in range(0, len(nb_attr)):
            train_by_attr.append([])
        for i in range(0, len(train)):
            for j in range(0, len(nb_attr)):
                train_by_attr[j].append(train[i][j])
        return train_by_attr


    def entropy(self, p):
        return - p * np.log2(p) - (1 - p)*np.log2((1 - p))

    def classification_error(p):
        return 1 - np.max([p, 1 - p])

    def choose_attribute(examples, attrs):
        best_attribute
        for i in range(0, len(attrs)):
            if (attrs[i]):

        return best_attribute

    def split_by_value(self, train):
        split = {}
        for value in train:
            split[value] = []

    #return true si on a un seul label
    def same_class(self, labels):
        return True if len(set(labels)) == 1 else False

    #Pour savoir value d'une colone en particulier
    def get_unique_value(self, rows, col):
        return set([row[col] for row in rows])

    def label_counts(self, labels):
        counts = {}
        for label in labels:
            if label is not counts.keys():
                counts[label] = 1
            counts[label] += 1
        return counts

class Node:

    def __init__(self, attr, attrname=None, default_child=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attrname = attrname or attr
        self.default_child = default_child
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attrvalue = example[self.attr]
        if attrvalue in self.branches:
            return self.branches[attrvalue](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch.  If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def __repr__(self):
        return ('DecisionFork({0!r}, {1!r}, {2!r})'
                .format(self.attr, self.attrname, self.branches))


class Leaf:

    def __init__(self, result):
        self.result = result


    # Vous pouvez rajouter d'autres methodes et fonctions,
    # il suffit juste de les commenter.
