import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import make_circles
from scipy.sparse import csgraph
from scipy.spatial import distance
import csv, time


def test_self():
    data_A, data_B = generate_data(10, 20)
    label_A = [data_A[0]]
    label_B = [data_B[-1]]
    unlabeled = data_A[1:] + data_B[:-1]
    # unlabeled.append((-5, -5))
    # unlabeled.append((0, 0))
    # unlabeled.append((5, 5))
    ti = time.clock()
    nn_self(label_A, label_B, unlabeled)
    te = time.clock()
    print('{}s'.format(te - ti))


def test_self_with_noise():
    data_A, data_B = generate_data(10, 20)
    label_A = [data_A[0]]
    label_B = [data_B[-1]]
    unlabeled = data_A[1:] + data_B[:-1]
    # noise added
    unlabeled.append((-5, -5))
    unlabeled.append((0, 0))
    unlabeled.append((5, 5))
    nn_self(label_A, label_B, unlabeled)


def generate_data(n, std):
    np.random.seed(999)
    data_A = [(x, y) for x in np.random.normal(-50, std, n) for y in np.random.normal(-50, std, n)]
    data_B = [(x, y) for x in np.random.normal(50, std, n) for y in np.random.normal(50, std, n)]
    return data_A, data_B


def nn_self(label_A, label_B, unlabeled):
    plt.figure(figsize=(8.5, 4))
    draw_self(label_A, label_B, unlabeled, 1)
    while unlabeled:
        dist_A = [{'from': A, 'to': item, 'dist': distance.euclidean(A, item)} for A in label_A for item in unlabeled]
        dist_B = [{'from': B, 'to': item, 'dist': distance.euclidean(B, item)} for B in label_B for item in unlabeled]
        min_A = min(dist_A, key=lambda x: x['dist'])
        min_B = min(dist_B, key=lambda x: x['dist'])
        if min_A['dist'] < min_B['dist']:
            label_A.append(min_A['to'])
            unlabeled.remove(min_A['to'])
        else:
            label_B.append(min_B['to'])
            unlabeled.remove(min_B['to'])

    draw_self(label_A, label_B, unlabeled, 2)
    plt.show()
    return label_A, label_B


def draw_self(label_A, label_B, unlabeled, opt):
    plt.subplot(1, 2, opt)
    plt.scatter([p[0] for p in label_A], [p[1] for p in label_A], color='navy',
                marker='s', lw=0, label="label A", s=10)
    plt.scatter([p[0] for p in label_B], [p[1] for p in label_B], color='c',
                marker='s', lw=0, label='label B', s=10)
    if unlabeled:
        plt.scatter([p[0] for p in unlabeled], [p[1] for p in unlabeled], color='darkorange',
                    marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    if unlabeled:
        plt.title("Before")
    else:
        plt.title("After")


def knn(X, labels):
    # #############################################################################
    # Learn with LabelSpreading
    label_spread = LabelSpreading(kernel='knn', alpha=0.6, max_iter=100)
    label_spread.fit(X, labels)

    # #############################################################################
    # Plot output labels
    output_labels = label_spread.transduction_

    return output_labels


def draw(X, labels, output_labels):
    label_0, label_1, unlabeled = 0, 1, -1

    plt.figure(figsize=(8.5, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X[labels == label_0, 0], X[labels == label_0, 1], color='navy',
                marker='s', lw=0, label="0 labeled", s=10)
    plt.scatter(X[labels == label_1, 0], X[labels == label_1, 1], color='c',
                marker='s', lw=0, label='1 labeled', s=10)
    plt.scatter(X[labels == unlabeled, 0], X[labels == unlabeled, 1], color='darkorange',
                marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Before")

    plt.subplot(1, 2, 2)
    output_label_array = np.asarray(output_labels)
    label_0_numbers = np.where(output_label_array == label_0)[0]
    label_1_numbers = np.where(output_label_array == label_1)[0]
    plt.scatter(X[label_0_numbers, 0], X[label_0_numbers, 1], color='navy',
                marker='s', lw=0, s=10, label="0 learned")
    plt.scatter(X[label_1_numbers, 0], X[label_1_numbers, 1], color='c',
                marker='s', lw=0, s=10, label="1 learned")
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("After")
    # plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)


def parse(file):
    X, labels = list(), list()
    with open(file, 'r') as f:
        dr = csv.DictReader(f)
        for row in dr:
            X.append((float(row['V1']), float(row['V2'])))
            if row['V3'].strip() == '':
                labels.append(-1)
            else:
                labels.append(int(row['V3']) - 1)

    return np.array(X), np.array(labels)


def run(dataset, label_rate):
    X, labels = parse('../../data/{}/tr_{}.csv'.format(dataset, label_rate))
    output_labels = knn(X, labels)
    draw(X, labels, output_labels)
    plt.show()


if __name__ == '__main__':
	# run dataset, label rate
	run('Data1', 0.02)
    
    pass
