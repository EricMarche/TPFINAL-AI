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
import matplotlib.pyplot as plt
import timeit
# le nom de votre classe
# NeuralNet pour le modele Reseaux de Neurones
# DecisionTree le modele des arbres de decision

class NeuralNet: #nom de la class a changer

    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre parametres au besoin,
        c'est a vous d'utiliser vos propres notations
        """
        self.w = []
        self.epoch = 20000
        self.best_dimension = 0
        self.best_layers = 0

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

        k = 6
        dimensions = [5, 10, 20, 30, 40, 50]
        hidden_layers = 1
        self.best_dimension = self.cross_validation(train, train_labels, k, dimensions, hidden_layers)
        print "dimension to choose : ", self.best_dimension

        self.weigths = []
        k = 5
        hidden_layers = [1, 2, 3, 4, 5]
        self.best_layers = self.cross_validation(train, train_labels, k, self.best_dimension, hidden_layers)
        print "best number of hidden layers : ", self.best_layers

        self.weigths = []
        RN_NON_ZERO = (self.neural_network(train, train_labels,
            self.best_dimension, self.best_layers, 0.1, test=True))
        test1 = [round(x) for x in RN_NON_ZERO]
        confusion_matrix(test1, train_labels)

        self.weigths = []
        RN_ZERO = (self.neural_network(train, train_labels,
            self.best_dimension, self.best_layers, 0.1, random=False, test=True))
        test2 = [round(x) for x in RN_ZERO]
        confusion_matrix(test2, train_labels)

        start = timeit.default_timer()
        result = self.neural_network(train, train_labels, self.best_dimension, self.best_layers, 0.1)
        prediction = [round(x) for x in result]
        confusion_matrix(prediction, train_labels)
        stop = timeit.default_timer()
        print "execution time : ",stop - start

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
        # predictions = []
        # start = timeit.default_timer()
        # for i in range(0, len(test)):
        #     prediction = self.predict(test[i], test_labels[i])
        #     predictions.append(prediction)
        # confusion_matrix(predictions, test_labels)
        # stop = timeit.default_timer()
        # print "execution time : ",stop - start

        # print "test labels : ", test_labels
        predictions = []
        start = timeit.default_timer()
        self.epoch = 1
        # test_result = (self.neural_network(test, test_labels,
        #     self.best_dimension, self.best_layers, 0.1, random=False))
        test_result = (self.neural_network(test, test_labels, 30, 3, 0.1))
        prediction = [round(x) for x in test_result]
        confusion_matrix(prediction, test_labels)
        stop = timeit.default_timer()
        print "execution time : ",stop - start

    def neural_network(self, train, labels, nb_neuronnes, nb_hidden_layer, learning_rate, random=True, test=False):

        weights = []
        if not test:
            weights = self.w

        np.random.seed(1)
        random_const_1 = 2
        random_const_2 = 1
        if not random:
            random_const_1 = 0
            random_const_2 = 0

        #Poid entre -1 et 1
        weight_input = random_const_1 * np.random.random((len(train[0]), nb_neuronnes)) \
            - random_const_2
        weight_output = random_const_1 * np.random.random((nb_neuronnes, 1)) \
            - random_const_2
        weights.append(weight_input)

        for i in range(nb_hidden_layer - 1):
            weights.append(random_const_1 * np.random.random((nb_neuronnes, nb_neuronnes)) \
                - random_const_2)

        weights.append(weight_output)

        for k in xrange(self.epoch):
            layers = []
            deltas = []
            i = 0
            #Boucle pour les layers (en ordre croissant)
            layers.append(train)
            for weight in weights:
                layers.append(self.sigmoid(np.dot( layers[i], weight)))
                i += 1

            i = 0
            #On fait l'operation a l'exterieur car differente des autres
            error = labels - layers[-1]
            #On ajoute les deltas dans l'ordre inverse
            deltas.append(error * self.sigmoid(layers[-1], deriv=True))

            #len(weights) - 1 pcq on a fait une operation a l'exterieur de la boucle
            for j in range(len(weights) - 1, 0, -1):
                error = deltas[i].dot(weights[j].T)
                deltas.append(error * self.sigmoid(layers[j], deriv=True))
                i += 1

            i = len(layers) - 1

            #update weigths
            for j in range(0, len(weights)):
                i -= 1
                weights[i] += layers[i].T.dot(learning_rate * deltas[j])

        return layers[-1]

    def sigmoid(self, x, deriv=False):
        if (deriv == True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    #Dimensions et hidden_layers sont des liste de k element pour que le code soit reutilisable
    def cross_validation(self, train, labels, k, dimensions, hidden_layers):
        best_value = 0
        best_prediction = 0
        n = len(train)
        # Pour dessiner notre graphique
        x = []
        y = []
        for i in range(0, k):
            split_train, split_test = fold_split(train, i * (n / k), (i + 1) * (n / k))
            split_train_labels, split_test_labels = fold_split(labels, i * (n / k), (i + 1) * (n / k))

            x_, y_ = self.cross_validation_train(split_train, split_train_labels,
             split_test_labels, dimensions, hidden_layers, i)
            if (y_ > best_prediction):
                best_prediction = y_
                best_value = x_
            x.append(x_)
            y.append(y_)
        plt.plot(x, y)
        # plt.show()
        return best_value

    def cross_validation_train(self, split_train, split_train_labels, split_test_labels,
     dimensions, hidden_layers, i):
        #Si on a une liste comme dimensions sa veut dire que on check pour la meilleur dimension
        #Sinon on cherche le meilleur layout
        self.weigths = []
        if type(dimensions) is list :
            result = self.neural_network(split_train, split_train_labels,
             dimensions[i], hidden_layers, 0.1, test=True)
            predict = prediction(result, split_train_labels)
            return dimensions[i], predict
        else:
            result= self.neural_network(split_train, split_train_labels,
             dimensions, hidden_layers[i], 0.1, test=True)
            predict = prediction(result, split_train_labels)
            return hidden_layers[i], predict

def fold_split(data, start, end):
    start = int(start)
    end = int(end)

    train = np.concatenate((data[:start], data[end:]))
    test = data[start:end]
    return train, test

def prediction(predict, labels):
    right = 0
    for i in range(0, len(labels)):
        if labels[i][0] == round(predict[i][0], 2):
            right += 1
    return (float(right)/float(len(labels)))

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

    # Vous pouvez rajouter d'autres methodes et fonctions,
    # il suffit juste de les commenter.
