import numpy as np

def sigmoid(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


X = np.array([[0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])
print "test : ", len(X[0])
np.random.seed(1)
syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

print "syn0 : "
print syn0
print "syn1 : "
print syn1
for j in xrange(1):

    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))


    print "l0"
    print l0
    print "l1"
    print l1
    print "l2"
    print l2
    l2_error = y - l2

    # if (j % 10000) == 0:
    #     print "Error:" + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    print "l2_error"
    print l2_error
    print "l2_delta"
    print l2_delta

    print "l1_error"
    print l1_error
    print "l1_delta"
    print l1_delta


    print "l1.T.dot(l2_delta)"
    print l1.T.dot(l2_delta)
    print ""
    print l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "output after training "
print l2
