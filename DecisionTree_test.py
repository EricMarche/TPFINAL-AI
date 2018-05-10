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

        attrs = []
        for i in range(0, len(train)):
            attrs.append(i)

        test = tree(train, train_labels, attrs)

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

def tree(train, labels, attr):
    labels = [ex[-1] for ex in train]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(train[0]) == 1:
        return majority(labels)
    bestFeat = choose(train)
    bestFeatLabel = attr[bestFeat]
    theTree = {bestFeatLabel:{}}
    del(attr[bestFeat])
    featValues = [ex[bestFeat] for ex in train]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subattr = attr[:]
        theTree[bestFeatLabel][value] = tree(split(train, bestFeat, value),subattr)
    return theTree

def majority(labels):
    classCount={}
    for vote in labels:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def choose(data):
    features = len(data[0])
    baseEntropy = entropy(data)
    bestInfoGain = 0.0;
    bestFeat = -1
    for i in range(features):
        featList = [ex[i] for ex in data]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            newData = split(data, i, value)
            probability = len(newData)/float(len(data))
            newEntropy += probability * entropy(newData)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat


def split(data, axis, val):
    newData = []
    for feat in data:
        if feat[axis] == val:
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis+1:])
            newData.append(reducedFeat)
    return newData

def entropy(data):
    entries = len(data)
    labels = {}
    for feat in data:
        label = feat[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels:
        probability = float(labels[key])/entries
        entropy -= probability * np.log(probability,2)
    return entropy
