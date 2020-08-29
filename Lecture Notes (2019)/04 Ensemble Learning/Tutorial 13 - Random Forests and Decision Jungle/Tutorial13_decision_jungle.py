import random
from collections import Counter
import numpy as np
import copy

def split_data(data, label=0, length=50):
    strings = [data[i:i+length] for i in range(0, len(data) - length, length)]
    random.shuffle(strings)
    strings = [(s, label) for s in strings]

    test = strings[:len(strings) * 10 // 100]
    training = strings[len(strings) * 10 // 100:]
    return test, training


def entropy(data):
    v = Counter([b for _, b in data]).values()
    v=list(v)
    d = np.array(v) / (sum(v))
    return - sum(d * np.log(d))


def split(train, feat):
    Hx = entropy(train)
    if Hx < 0.000001:
        raise Exception("Entropy very low")
    L1 = []
    L2 = []
    for t in train:
        if feat in t[0]:
            L1 += [t]
        else:
            L2 += [t]

    E1 = entropy(L1)
    E2 = entropy(L2)
    L = float(len(train))

    H = Hx - E1 * len(L1)/L - E2 * len(L2)/L
    return H, L1, L2, feat

## ----------------------------
## - The decision jungle code -
## ----------------------------

def build_jungle(train, features, levels=20, numfeatures=100):
    DAG = {0: copy.copy(train)}
    Candidate_sets = [0]
    next_ID = 0
    M = 20

    for level in range(levels):
        result_sets = []
        for tdata_idx in Candidate_sets:
            tdata = DAG[tdata_idx]

            if entropy(tdata) == 0.0:
                next_ID += 1
                idx1 = next_ID
                result_sets += [idx1]
                DAG[idx1] = tdata + []
                del DAG[tdata_idx][:]
                DAG[tdata_idx] += [True, idx1, idx1]
                continue

            X = (split(tdata, F) for F in random.sample(features, numfeatures))
            H, L1, L2, F = max(X)

            # Branch = (F, M1, M2)
            next_ID += 1
            idx1 = next_ID
            DAG[idx1] = L1
            next_ID += 1
            idx2 = next_ID
            DAG[idx2] = L2

            result_sets += [idx1, idx2]
            del DAG[tdata_idx][:]
            DAG[tdata_idx] += [F, idx1, idx2]

        ## Now optimize the result sets here
        random.shuffle(result_sets)

        basic = result_sets[:M]
        for r in result_sets[M:]:
            maxv = None
            maxi = None
            for b in basic:
                L = float(len(DAG[r] + DAG[b]))
                e1 = len(DAG[r]) * entropy(DAG[r])
                e2 = len(DAG[b]) * entropy(DAG[b])
                newe = L * entropy(DAG[r] + DAG[b])
                score = abs(e1 + e2 - newe)
                if maxv is None:
                    maxv = score
                    maxi = b
                    continue
                if score < maxv:
                    maxv = score
                    maxi = b
            DAG[maxi] += DAG[r]
            del DAG[r]
            DAG[r] = DAG[maxi]

        Candidate_sets = basic

    for tdata_idx in Candidate_sets:
        tdata = DAG[tdata_idx]
        C1 = Counter([b for _, b in tdata])
        del DAG[tdata_idx][:]
        DAG[tdata_idx] += [None, C1]

    return DAG


def classify_jungle(DAG, item):
    branch = DAG[0]
    while branch[0] is not None:
        try:
            fet, L1, L2 = branch
            if fet == True or fet in item:
                branch = DAG[L1]
            else:
                branch = DAG[L2]
        except:
            print (len(branch))
            raise
    return branch[1]

## -------------------------
## - Sample classification -
## -------------------------

if __name__ == "__main__":

    dataEN = open("C:\\Users\hc\PycharmProjects\BA_dt\data\origin\pg23428.txt").read()
    dataFR = open("C:\\Users\hc\PycharmProjects\BA_dt\data\origin\pg5711.txt").read()

    length = 200

    testEN, trainEN = split_data(dataEN, label=0, length=length)
    testFR, trainFR = split_data(dataFR, label=1, length=length)

    print ("training: EN=%s FR=%s" % (len(trainEN), len(trainFR)))

    train = trainEN + trainFR
    random.shuffle(train)
    test = testEN + testFR
    random.shuffle(test)

    sometrain = random.sample(train, 1000)
    features = set()
    while len(features) < 700:
        fragment, _ = random.choice(sometrain)
        l = int(round(random.expovariate(0.20)))
        b = random.randint(0, max(0, length - l))
        feat = fragment[b:b+l]

        ## Test
        C = 0
        for st, _ in sometrain:
            if feat in st:
                C += 1

        f = float(C) / 1000
        if f > 0.01 and f < 0.99 and feat not in features:
            features.add(feat)

    features = list(features)

    jungle = []
    for i in range(10):
        print ("Build tree %s" % i)
        size = len(train) / 3
        size = int(size)
        training_sample = random.sample(train, size)

        tree = build_jungle(training_sample, features, numfeatures=100)
        jungle += [tree]

    testdata = test

    results_jungle = Counter()
    for item, cat in testdata:
        # Jungle
        c = Counter()
        for tree in jungle:
            c += classify_jungle(tree, item)
        res = (max(c, key=lambda x: c[x]), cat)
        results_jungle.update([res])

    print
    print ("Results            Jungle")
    print ("True positives:     %4d" % (results_jungle[(1, 1)]))
    print ("True negatives:     %4d" % (results_jungle[(0, 0)]))
    print ("False positives:    %4d" % (results_jungle[(1, 0)]))
    print ("False negatives:    %4d" % (results_jungle[(0, 1)]))
