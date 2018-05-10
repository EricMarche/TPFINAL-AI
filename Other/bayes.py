import numpy as np
data_list = [
 [0, 85, 85, 0],
 [0, 80, 90, 1],
 [1, 83, 86, 0],
 [2, 70, 96, 0],
 [2, 68, 80, 0],
 [2, 65, 70, 1],
 [1, 64, 65, 1],
 [0, 72, 95, 0],
 [0, 69, 70, 0],
 [2, 75, 80, 0],
 [0, 75, 70, 1],
 [1, 72, 90, 1],
 [1, 81, 75, 0],
 [2, 71, 91, 1]]

labels_list = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
data = np.array(data_list).astype(np.float)
labels = np.array(labels_list)


def bayes( train, train_labels):
 means = {}
 variances = {}
 #Nous permet d avoir les type de labels
 group = set(train_labels)
 for label in group:
     data_by_label = train[train_labels == label]
     print "data_by_label : ", data_by_label
     means[label] = data_by_label.mean(axis=0)
     variances[label] = data_by_label.var(axis=0)
 return means, variances

#On y passe un vector exemple
def probability(means, variances, group, exemple):
 probabilities = {}
 for label in group:
     probabilities[label] = 1
     for i in range(0, len(means[label])):
         part_1 = 1/ (np.sqrt(2 * np.pi) * variances[label][i])
         part_2 = (np.power((exemple[i] - means[label][i]), 2) * -1)/ (2 *np.power(variances[label][i],2))
         result =  part_1 * np.exp(part_2)

         probabilities[label] *= result

 return probabilities

means, variances = bayes(data, labels)
group = set(labels)
test = [1, 70, 62, 1]
probabilities = probability(means, variances, group, test)

print "probabilities: ", probabilities

