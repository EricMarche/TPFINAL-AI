import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont etre attribues a l entrainement,
        le rest des exemples va etre utilise pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilise
        pour lentrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilises pour l entrainement, chaque
        ligne dans cette matrice represente un exemple (ou instance) d entrainement.

        - train_labels : contient les labels (ou les etiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l etiquette) pour l exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilises pour le test, chaque
        ligne dans cette matrice represente un exemple (ou instance) de test.

        - test_labels : contient les labels (ou les etiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l etiquette) pour l exemple test[i]
    """

    random.seed(1) # Pour avoir les meme nombres aleatoires a chaque initialisation.

    # Vous pouvez utiliser des valeurs numeriques pour les differents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

    # Le fichier du dataset est dans le dossier datasets en attache
    f = open('datasets/bezdekIris.data', 'r')
    data = f.readlines()
    random.shuffle(data)

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []
    items = []
    for lines in data:
        words = lines.split(',')
        items.append(words)

    # TODO : le code ici pour lire le dataset

    # REMARQUE tres importante :
    # remarquez bien comment les exemples sont ordonnes dans
    # le fichier du dataset, ils sont ordonnes par type de fleur, cela veut dire que
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc a utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de separer
    # en train et test.


    number_of_train = int(len(items) * train_ratio)

    for x in range(0, number_of_train):
        train_list.append([items[x][0], items[x][1], items[x][2], items[x][3]])
        train_labels_list.append([conversion_labels[items[x][4].strip()]])

    for x in range(number_of_train, len(items)):
        test_list.append([items[x][0], items[x][1], items[x][2], items[x][3]])
        test_labels_list.append([conversion_labels[items[x][4].strip()]])

    train = np.array(train_list).astype(np.float)
    train_labels = np.array(train_labels_list).astype(np.float)
    test = np.array(test_list).astype(np.float)
    test_labels = np.array(test_labels_list).astype(np.float)

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)



def load_congressional_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Congressional Voting Records

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilise pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilises pour l'entrainement, chaque
        ligne dans cette matrice represente un exemple (ou instance) d'entrainement.

        - train_labels : contient les labels (ou les etiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilises pour le test, chaque
        ligne dans cette matrice represente un exemple (ou instance) de test.

        - test_labels : contient les labels (ou les etiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """

    random.seed(1) # Pour avoir les meme nombres aleatoires a chaque initialisation.

    # Vous pouvez utiliser un dictionnaire pour convertir les attributs en numeriques
    # Notez bien qu'on a traduit le symbole "?" pour une valeur numerique
    # Vous pouvez biensur utiliser d'autres valeurs pour ces attributs
    conversion_labels = {'republican' : 0, 'democrat' : 1,
                         'n' : 0, 'y' : 1, '?' : 2}

    # Le fichier du dataset est dans le dossier datasets en attache
    f = open('datasets/house-votes-84.data', 'r')
    data = f.readlines()
    random.shuffle(data)

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []
    items = []
    for lines in data:
        words = lines.split(',')
        items.append(words)

    # TODO : le code ici pour lire le dataset
    number_of_train = int(len(items) * train_ratio)

    for x in range(0, number_of_train):
        train_labels_list.append([conversion_labels[items[x][0]]])
        row = []
        for y in range(1, len(items[x]) - 1):
            row.append(conversion_labels[items[x][y].strip()])
        train_list.append(row)

    for x in range(number_of_train, len(items)):
        test_labels_list.append([conversion_labels[items[x][0]]])
        row = []
        for y in range(1, len(items[x]) - 1):
            row.append(conversion_labels[items[x][y].strip()])
        test_list.append(row)

    train = np.array(train_list).astype(np.float)
    train_labels = np.array(train_labels_list).astype(np.float)
    test = np.array(test_list).astype(np.float)
    test_labels = np.array(test_labels_list).astype(np.float)

    # La fonction doit retourner 4 structures de donnees de type Numpy.
    return (train, train_labels, test, test_labels)


def load_monks_dataset(numero_dataset):
    """Cette fonction a pour but de lire le dataset Monks

    Notez bien que ce dataset est different des autres d'un point de vue
    exemples entrainement et exemples de tests.
    Pour ce dataset, nous avons 3 differents sous problemes, et pour chacun
    nous disposons d'un fichier contenant les exemples d'entrainement et
    d'un fichier contenant les fichiers de tests. Donc nous avons besoin
    seulement du numero du sous probleme pour charger le dataset.

    Args:
        numero_dataset: lequel des sous problemes nous voulons charger (1, 2 ou 3 ?)
		par exemple, si numero_dataset=2, vous devez lire :
			le fichier monks-2.train contenant les exemples pour l'entrainement
			et le fichier monks-2.test contenant les exemples pour le test
        les fichiers sont tous dans le dossier datasets
    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy, train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilises pour l'entrainement, chaque
        ligne dans cette matrice represente un exemple (ou instance) d'entrainement.
        - train_labels : contient les labels (ou les etiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est le label (ou l'etiquette) pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilises pour le test, chaque
        ligne dans cette matrice represente un exemple (ou instance) de test.
        - test_labels : contient les labels (ou les etiquettes) pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est le label (ou l'etiquette) pour l'exemple test[i]
    """
    train_file_name = 'datasets/monks-%s.train'% (numero_dataset)
    test_file_name = 'datasets/monks-%s.test'% (numero_dataset)
    train_file = open(train_file_name, 'r')
    test_file = open(test_file_name, 'r')
    data = train_file.readlines()

    train_list = []
    train_labels_list = []
    test_list = []
    test_labels_list = []
    for lines in data:
        words = lines.split()
        train_labels_list.append([words.pop(0)])
        # pour le moment on pop l'id pcq il est pas utile et parsable en float
        words.pop()
        train_list.append(words)

    data = test_file.readlines()
    for lines in data:
        words = lines.split()
        # pour le moment on pop l'id pcq il est pas utile et parsable en float
        words.pop()
        test_labels_list.append([words.pop(0)])
        test_list.append(words)

    # train = np.array(train_list).astype(np.float)
    # train_labels = np.array(train_labels_list).astype(float)
    # test = np.array(test_list).astype(np.float)
    # test_labels = np.array(test_labels_list).astype(float)

    train = np.array(train_list)
    train_labels = np.array(train_labels_list)
    test = np.array(test_list)
    test_labels = np.array(test_labels_list)

    # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)
