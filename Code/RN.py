"""
Vous allez definir une classe pour chaque algorithme que vous allez developper,
votre classe doit contenit au moins les 3 methodes definies ici bas,
	* train 	: pour entrainer le modele sur l'ensemble d'entrainement
	* predict 	: pour predire la classe d'un exemple donne
	* test 		: pour tester sur l'ensemble de test
vous pouvez rajouter d'autres methodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les methodes train, predict et test de votre code.
"""

import numpy as np


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
        biais = 1
        #synapses
        self.syn1 = 2*np.random.random((3,4)) - biais
        self.syn2 = 2*np.random.random((2,3)) - biais
        self.syn3 = 2*np.random.random((4,1)) - biais
        self.syn4 = 2*np.random.random((2,1)) - biais
        self.syn5 = 2*np.random.random((4,2)) - biais


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

        for i in xrange(6000):
            #layers
            layer_input = train
            #np.dot multiplication matricielle
            layer_1 = nonlinear(np.dot(layer_input, self.syn1))
            layer_2 = nonlinear(np.dot(layer_1, self.syn2))
            layer_3 = nonlinear(np.dot(layer_2, self.syn3))
            layer_4 = nonlinear(np.dot(layer_3, self.syn4))
            layer_5 = nonlinear(np.dot(layer_4, self.syn5))

            #backpropagation
            layer5_error = test_labels - layer_5
            #On fait juste afficher pour nous donne une idee
            if (i % 10000) == 0:
                print "error : " + str(np.mean(np.abs(layer5_error)))

            layer5_delta = layer5_error*nonlinear(layer_5, deriv=True)

            layer4_error = layer5_delta.dot(syn5.T)
            layer4_delta = layer4_error * nonlinear(layer_4, deriv=True)

            layer4_error = layer5_delta.dot(syn5.T)
            layer4_delta = layer4_error * nonlinear(layer_4, deriv=True)

            layer3_error = layer5_delta.dot(syn5.T)
            layer3_delta = layer4_error * nonlinear(layer_4, deriv=True)

            layer2_error = layer5_delta.dot(syn5.T)
            layer2_delta = layer4_error * nonlinear(layer_4, deriv=True)

            layer1_error = layer5_delta.dot(syn5.T)
            layer1_delta = layer4_error * nonlinear(layer_4, deriv=True)

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
    def kfold_crossvalidation(self, train, test) {
        k = 5
        for i in range(0, 5):

    }

	def nonlinear(x, deriv=False):
        if deriv:
            return x*(x-1)
        return 1/(1+np.exp(-x))

	# Vous pouvez rajouter d'autres methodes et fonctions,
	# il suffit juste de les commenter.
