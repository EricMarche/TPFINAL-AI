import numpy as np

def threshold(value):
    return (1 / (1 + np.exp(-value)))
a = 0.5
w = [[2, 1, -1]]
matrice_w = np.array(w).astype(np.int)

x1 = [[1],
     [-1],
     [1]]
x2 = [[0],
      [1],
      [-1]]
x3 = [[1],
      [1],
      [1]]

x = [[1, -1, 1],
     [0, 1, -1],
     [1, 1, 1]]


w_prime = [[2.7311, 2.0483, -2.0505]]
biais = 0.0483


y1 = np.dot(w, x[1]) + biais
print "y : ", y1
print "z : ", threshold(y1)

# y = [1, 0, 1]
# matrice_x = np.array(x).astype(np.int)
# b = 1
# for j in range(10):
#     for i in range(0, 3):
#         produit = np.dot(w, x[i])
#         #+1 c'est le biais
#         result = threshold(produit + b)
#
#         # print "produit : ", produit
#         # print "result : ", result
#         # print "expected : ", y[i]
#         if result != y[i]:
#             w = w + a * (y[i] - result) * x[i]
#             # print " nouveau w : ", w
#             b = b + a * (y[i] - result)
#             # print " nouveau b : ", b
#         else:
#             print "b :", b
#             print "w : ", w
#     # print "************"
#     # print "test : ", threshold(0)
#     # if (result != )
#
#     # )
