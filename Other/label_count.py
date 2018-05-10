

def label_counts(data):
    counts = {}
    for row in data:
        label = row[-1]
        if label is not counts.keys():
            counts[label] = 1
        counts[label] += 1
    return counts

countdict = label_counts([[1, 2], [1, 2], [1, 4]])
print "nombre de donnees : ", len(countdict)
