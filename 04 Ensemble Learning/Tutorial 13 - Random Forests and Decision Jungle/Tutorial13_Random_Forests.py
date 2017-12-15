
# coding: utf-8

# In[14]:


from collections import Counter
from time import time
import numpy as np


# In[15]:


def evaluate_performance(trials=1):
    filename = 'C:/data/random_forest_data.txt'
    data = np.loadtxt(filename, delimiter=',')  # (267,45)
    X = data[:, 1:]
    y = data[:, 0]
    n, d = X.shape  # n = 행 , d = 열 (267,44)
    print('input_data\n', X)
    print('label_data\n', y)
    print('input_data_set_shape', n, d)
    print('label_data_set_shape', n, d)

    accuracy_list = np.zeros(trials)

    for trial in range(trials):
        print('##    Trial    ##', trial)

        # Random shuffle
        idx = np.arange(n)
        print('idx\n', idx)  # idx list
        np.random.seed(np.int(time() / 150))
        np.random.shuffle(idx)  # X, y data set shuffle
        X = X[idx]  # shuffled input data
        Y = y[idx]

        # Split Train and Test samples( train(0.8) / test(0.2)
        train = np.int(0.8 * n)
        Xtrain = X[:train, :]  # (213,44)
        Xtest = X[train:, :]
        Ytrain = Y[:train]
        Ytest = Y[train:]
        print('Xtran\n', Xtrain)
        print('Ytrain\n', Ytrain)

        tr_n, tr_d = Xtrain.shape  # check train data shape
        te_n, te_d = Xtest.shape

        print('Xtrain_shape : ', tr_n, tr_d)
        print('Xtest_shape : ', te_n, te_d)

#         # setting parameter for make forest
        clf = RandomForest(n_trees=3, max_depth=10, ratio_per_tree=0.7, ratio_features=0.3)
        # clf = RandomForest(n_trees=15, max_depth=100, ratio_per_tree=0.7, ratio_features=0.3)
        clf.fit(Xtrain, Ytrain)

        print('')
        print('##    Test Start    ##\n')

        pred = clf.predict(Xtest)

        accuracy_list[trial] = accuracy_score(Ytest, pred)


    stats = np.zeros((3, 3))
    stats[0, 0] = np.mean(accuracy_list)
#     stats[0, 1] = np.std(accuracy_list)

    return stats


# input_data
#  [[ 59.  52.  70. ...,  74.  64.  67.]
#  [ 72.  62.  69. ...,  71.  56.  58.]
#  [ 71.  62.  70. ...,  41.  51.  46.]
#  ..., 
#  [ 75.  73.  72. ...,  75.  67.  71.]
#  [ 59.  62.  72. ...,  76.  70.  70.]
#  [ 64.  66.  68. ...,  64.  57.  54.]]
# label_data
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# input_data_set_shape 267 44
# label_data_set_shape 267 44
# 
# idx
#  [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
#   18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
#   36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
#   54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
#   72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
#   90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
#  108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
#  126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
#  144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
#  162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
#  180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
#  198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
#  216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
#  234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
#  252 253 254 255 256 257 258 259 260 261 262 263 264 265 266]
# Xtran
#  [[ 32.  41.  76. ...,   8.  18.  11.]
#  [ 67.  80.  73. ...,  67.  58.  56.]
#  [ 64.  53.  74. ...,  67.  71.  67.]
#  ..., 
#  [ 58.  63.  80. ...,  77.  65.  66.]
#  [ 56.  68.  58. ...,  64.  52.  54.]
#  [ 73.  76.  68. ...,  30.  15.  11.]]
# Ytrain
#  [ 1.  1.  0.  0.  1.  1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  0.  1.  1.  1.  1.  0.  0.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  0.  0.  0.  1.  1.  0.  0.  1.
#   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.
#   1.  0.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  0.  1.  1.  1.
#   1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  0.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  1.
#   1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  1.  0.  1.  1.  0.
#   1.  0.  1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  1.  1.  1.
#   1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.  1.  1.  0.
#   1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  1.  1.  0.  1.  1.]
# Xtrain_shape :  213 44
# Xtest_shape :  54 44

# In[16]:


# Make Forest
class RandomForest(object):
    def __init__(self, n_trees=10, max_depth=10, ratio_per_tree=0.5, ratio_features=1.0):
        self.trees = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.ratio_per_tree = ratio_per_tree
        self.ratio_features = ratio_features

    def fit(self, X, Y):
        n, d = X.shape
        print('Forest_data_shape', n, d)
        print('Num of tree : ', self.n_trees)
        self.trees = []  # contain tree

        # Maek tree
        for i in range(self.n_trees):
            idx = np.arange(n)
            # print(idx)
            # np.random.seed(np.int(time() / 100))
            np.random.shuffle(idx)
            X = X[idx]
            Y = Y[idx]
            train = np.int(self.ratio_per_tree * n)
            print('train(ratio_per_tree * n) : ', train)
            Xtrain = X[:train, :]
            print('tree_train_data', Xtrain)
            Ytrain = Y[:train]
            print('tree_label_train_data', Ytrain)
            
            clf = RandomTree(max_depth=self.max_depth, ratio_features=self.ratio_features)
            # print('for_tree_max_depth : ', self.max_depth)
            # print('for_tree_ratio_features : ', self.ratio_features)
            # print('forest_ratio_feaures', self.ratio_features)
            clf.fit(Xtrain, Ytrain)  # 객체 생성
            # print('forest_shape', Xtrain.shape)
            self.trees.append(clf)
            # print('tress', self.trees)
            
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        print('predictions 1 : ', predictions)
        predictions = np.array([np.argmax(np.bincount(prediction)) for prediction in predictions])
        print('predictions 2 : ', predictions)
        
        return predictions


# Forest_data_shape 213 44
# Num of tree :  1
# train(ratio_per_tree * n) :  149
# tree_train_data [[ 55.  54.  71. ...,  27.  29.  22.]
#  [ 73.  73.  75. ...,  67.  64.  61.]
#  [ 76.  75.  68. ...,  70.  63.  61.]
#  ..., 
#  [ 66.  70.  78. ...,  64.  58.  56.]
#  [ 74.  73.  72. ...,  65.  55.  56.]
#  [ 65.  66.  71. ...,  67.  53.  42.]]
# tree_label_train_data [ 1.  1.  0.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  0.  0.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  0.  0.  1.  0.  1.  1.  0.  1.  1.  0.  1.  0.  1.
#   1.  1.  0.  0.  1.  0.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.
#   0.  1.  1.  1.  0.  1.  1.  0.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.
#   0.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.
#   0.  1.  1.  1.  1.  1.  1.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.
#   1.  1.  1.  1.  1.  1.  0.  1.  1.  1.  1.  1.  0.  1.  1.  0.  0.  1.
#   0.  1.  1.  0.  1.]

# In[17]:


class DecisionNode(object):
    def __init__(self,
                 feature=None,
                 value=None,
                 false_branch=None,
                 true_branch=None,
                 is_leaf=False,
                 current_results=None
                 ):
        self.feature = feature
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf
        self.current_results = current_results


# In[18]:


class RandomTree(object):
    def __init__(self, max_depth=10, ratio_features=1.0):
        self.tree = None
        self.max_depth = max_depth
        self.ratio_features = ratio_features
        # self.get_split_feature(self)
        # self.split_feature_list = []

    def fit(self, X, Y):
        print('forest_ratio_feaures', self.ratio_features)
        Y = Y.reshape((1, X.shape[0])).T
        print('reshape_y', Y)
        data = np.concatenate((X, Y), axis=1)
        print('tree_data\n', data)
        print('tree_data_shape\n', data.shape)
        self.tree = build_tree(data, ratio_features=self.ratio_features, max_depth=self.max_depth)

    def predict(self, X):
        predict_show = np.array([pred(self.tree, x) for x in X])
        print('predict_show  : ', predict_show) # tree 개수 만큼 출력된다.
        
        return np.array([pred(self.tree, x) for x in X])


# reshape_y [[ 0.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 0.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 0.]
#  [ 1.]
#  [ 1.]
#  [ 1.]
#  [ 0.]]
# tree_data
#  [[ 72.  61.  64. ...,  56.  52.   0.]
#  [ 65.  69.  70. ...,  53.  55.   1.]
#  [ 62.  54.  65. ...,  27.  18.   1.]
#  ..., 
#  [ 69.  68.  75. ...,  63.  67.   1.]
#  [ 65.  56.  67. ...,  59.  65.   1.]
#  [ 70.  72.  70. ...,  66.  63.   0.]]
# 
# tree_data_shape
#  (149, 45)

# In[19]:


def label_frequencies(data):
    n, d = data.shape
    labels = data[:, (d - 1):].reshape(n).astype(int)
    return np.bincount(labels)


def different_labels(labels):
    c = 0
    for label in labels:
        if label != 0:
            c += 1
    return c


# In[20]:


def split_data(data, feature, value):  # data = tree data set(56,5), featuer : index, value : feature value
    t, f = [], []
    for row in data:
        if row[feature] >= value:
            t.append(row)
        else:
            f.append(row)

    return np.array(t), np.array(f)


# current_result :  [ 30 119]
# t
#  [[ 72.  61.  64. ...,  56.  52.   0.]
#  [ 65.  69.  70. ...,  53.  55.   1.]
#  [ 62.  54.  65. ...,  27.  18.   1.]
#  ..., 
#  [ 69.  68.  75. ...,  63.  67.   1.]
#  [ 65.  56.  67. ...,  59.  65.   1.]
#  [ 70.  72.  70. ...,  66.  63.   0.]]
# f
#  []

# In[21]:


def entropy(data):
    n = len(data)
    if n == 0:
        return 0
    labels = label_frequencies(data).astype(float) / n
    # print('Prb_label_select : ', labels)
    e = .0
    for label in labels:
        if label > .0:
            e -= label * np.log2(label)
#     print('labels_entropy : ', e)

    return e

def entropy_for_feature(data, feature):  # feature는 feature를 뽑은 index가 저장된다.
    values = np.unique(data[:, feature])  # values 에는 feature 값들이 저장된다.
    # print('entropy_feature_values\n', values)
    entropy_loss = 1e5
    value = 0
    for val in values:
        t, f = split_data(data, feature, val)
        # print('true : ', t)
        # print('false : ', f)
        # print('t_entropy', t)
        curr_entropy_loss = (len(t) * entropy(t) + len(f) * entropy(f)) / len(data)
        print('curr_entropy_loss : ', curr_entropy_loss)
        if entropy_loss > curr_entropy_loss: # entropy loss가 가장 작은 value를 추출한다.
            entropy_loss = curr_entropy_loss
            value = val
    print('entropy_for_featrue : ',entropy_loss)
    return entropy_loss, value  # entropy loss가 가장 작은 feature value 값과 entropy값 return


# curr_entropy_loss :  0.774839040857
# curr_entropy_loss :  0.772321475811
# curr_entropy_loss :  0.769784398225
# curr_entropy_loss :  0.7672275015
# curr_entropy_loss :  0.76465047171
# curr_entropy_loss :  0.762052987366
# curr_entropy_loss :  0.759434719169
# curr_entropy_loss :  0.756795329755
# curr_entropy_loss :  0.754134473428
# curr_entropy_loss :  0.75145179588
# curr_entropy_loss :  0.746019515075
# curr_entropy_loss :  0.743269157471
# curr_entropy_loss :  0.734876482828
# curr_entropy_loss :  0.723340127664
# curr_entropy_loss :  0.720391252051
# curr_entropy_loss :  0.714412410634
# curr_entropy_loss :  0.746554909397
# curr_entropy_loss :  0.75338906286
# curr_entropy_loss :  0.742807244048
# curr_entropy_loss :  0.742575652946
# curr_entropy_loss :  0.738023671973
# curr_entropy_loss :  0.730754083796
# curr_entropy_loss :  0.764485698308
# curr_entropy_loss :  0.755114685676
# curr_entropy_loss :  0.762259149622
# curr_entropy_loss :  0.768412475558
# curr_entropy_loss :  0.774183062847
# curr_entropy_loss :  0.773620870622
# curr_entropy_loss :  0.774832927238
# curr_entropy_loss :  0.771681698273
# curr_entropy_loss :  0.767502779496
# curr_entropy_loss :  0.771436041625
# curr_entropy_loss :  0.770650831554
# curr_entropy_loss :  0.773212772351
# curr_entropy_loss :  0.76465047171
# curr_entropy_loss :  0.772321475811
# entropy_for_featrue :  0.714412410634
# best_value_for_feature :  56.0
# 

# In[22]:


split_feature_list = []


def get_split_feature(fea):
    split_feature_list.append(fea)
    


    return split_feature_list


# In[23]:


def build_tree(data, depth=0, ratio_features=1.0, max_depth=100):
    n = len(data)
    if n == 0:
        return DecisionNode(is_leaf=True)
    if n != 0:
        n, d = data.shape

    current_results = label_frequencies(data)  # result : [ number_of_0 , unmber_of_1]
    if depth == max_depth:  # 나무의 depth가 max_depth에 도달했거나 하나의 클래스만 남았을 경우 노드를 종료한다.
        return DecisionNode(current_results=current_results, is_leaf=True)
    if different_labels(current_results) == 1:
        return DecisionNode(current_results=current_results, is_leaf=True)

    # Select Random Features
    seed = np.int(time() / 150)
    idx = np.arange(d - 1)
    np.random.seed(seed)
    np.random.shuffle(idx)
    idx = idx[: np.int(ratio_features * (d - 1))]
    print('selected_idx\n', idx)

    entropy_loss = 1e5
    feature = None
    value = None

    for feat in idx:
        print('chekc_feat : ', feat)
        entropy, val = entropy_for_feature(data, feat)
        if entropy_loss > entropy:
            entropy_loss = entropy
            feature = feat
            value = val
            split_feature = get_split_feature(feature) # importance variable을 구하기 위해 split에 사용된 feature를 저장한다.
            print('split_feature : ', split_feature)
            cnt = Counter(split_feature) 
            print('count_split_feature\n', cnt)

#         if len(split_feature) == np.int(ratio_features * (d - 1)) + 1: # tree개수 만큼 feature가 저장되면 초기화한다.
#             split_feature = []

    print('select_feature_list : ', split_feature_list)
    print('Best for split feature : ', feature)
    print('Best for split value : ', value)
    print('')
    print('###  real tree split start  ###')
    t, f = split_data(data, feature, value) # entropy loss가 가장 낮은 feature를 가지고 child node를 만들어 나간다.    
    print('false_branch_shape', f.shape)
    print('true_branch_shape', t.shape)


    return DecisionNode(

        feature=feature,
        value=value,
        false_branch=build_tree(f, depth + 1, ratio_features, max_depth),
        true_branch=build_tree(t, depth + 1, ratio_features, max_depth),
        current_results=current_results
    )


# selected_idx
#  [24 35 37 10  6 34 23 18 16  1 28 13 15]
# chekc_feat :  24
# curr_entropy_loss :  0.737745933572
# curr_entropy_loss :  0.735478793959
# curr_entropy_loss :  0.730892242388
# curr_entropy_loss :  0.728572287352
# curr_entropy_loss :  0.726234173374
# curr_entropy_loss :  0.723877612529
# curr_entropy_loss :  0.721502309952
# curr_entropy_loss :  0.719107963614
# curr_entropy_loss :  0.714260894315
# curr_entropy_loss :  0.706839472705
# curr_entropy_loss :  0.704324089216
# curr_entropy_loss :  0.701787326171
# curr_entropy_loss :  0.69922881489
# curr_entropy_loss :  0.694045024174
# curr_entropy_loss :  0.688769567814
# curr_entropy_loss :  0.677930187269
# curr_entropy_loss :  0.672359050856
# curr_entropy_loss :  0.666681794755
# curr_entropy_loss :  0.663802083859
# curr_entropy_loss :  0.657957785664
# curr_entropy_loss :  0.682786168338
# curr_entropy_loss :  0.674699620804
# curr_entropy_loss :  0.671926604332
# curr_entropy_loss :  0.66337092408
# curr_entropy_loss :  0.64829600186
# curr_entropy_loss :  0.63873605537
# curr_entropy_loss :  0.667063313831
# curr_entropy_loss :  0.673941856371
# curr_entropy_loss :  0.684643221517
# curr_entropy_loss :  0.673900434633
# curr_entropy_loss :  0.666925072953
# curr_entropy_loss :  0.688036752597
# curr_entropy_loss :  0.697790894159
# curr_entropy_loss :  0.703498942991
# curr_entropy_loss :  0.694776137972
# curr_entropy_loss :  0.696206007742
# curr_entropy_loss :  0.721708556897
# curr_entropy_loss :  0.737277922541
# curr_entropy_loss :  0.737688238021
# curr_entropy_loss :  0.730005110948
# curr_entropy_loss :  0.733707486359
# curr_entropy_loss :  0.725529483166
# curr_entropy_loss :  0.734715845834
# curr_entropy_loss :  0.735828034904
# curr_entropy_loss :  0.730892242388
# curr_entropy_loss :  0.735478793959
# entropy_for_featrue :  0.63873605537
# split_feature :  [24]
# count_split_feature
#  Counter({24: 1})
# chekc_feat :  35
# curr_entropy_loss :  0.737745933572
# curr_entropy_loss :  0.735478793959
# curr_entropy_loss :  0.733194319692
# curr_entropy_loss :  0.730892242388
# curr_entropy_loss :  0.728572287352
# curr_entropy_loss :  0.726234173374
# 
# :
# :
# :
# :
# curr_entropy_loss :  0.737534941232
# curr_entropy_loss :  0.716694264091
# curr_entropy_loss :  0.726234173374
# curr_entropy_loss :  0.730892242388
# curr_entropy_loss :  0.735478793959
# entropy_for_featrue :  0.632292357883
# split_feature :  [24, 15]
# count_split_feature
#  Counter({24: 1, 15: 1})
# select_feature_list :  [24, 15]
# Best for split feature :  15
# Best for split value :  60.0
# 
# --------------------------------------------------------------------
# false_branch_shape (22, 45)
# true_branch_shape (68, 45)
# selected_idx
#  [24 35 37 10  6 34 23 18 16  1 28 13 15]
# chekc_feat :  24
# curr_entropy_loss :  0.266764987803
# curr_entropy_loss :  0.263641090028
# curr_entropy_loss :  0.260360870105
# curr_entropy_loss :  0.256907851339
# curr_entropy_loss :  0.249403104603
# curr_entropy_loss :  0.245301866631
# curr_entropy_loss :  0.236238753317
# curr_entropy_loss :  0.231189378508
# curr_entropy_loss :  0.219748493461
# curr_entropy_loss :  0.213179815268
# curr_entropy_loss :  0.205878409681
# curr_entropy_loss :  0.236238753317
# curr_entropy_loss :  0.245301866631
# curr_entropy_loss :  0.256907851339
# curr_entropy_loss :  0.260360870105
# curr_entropy_loss :  0.263641090028
# entropy_for_featrue :  0.205878409681
# split_feature :  [24, 15, 24, 35, 37, 23, 24, 37, 6, 24, 24, 37, 1, 24]
# count_split_feature
#  Counter({24: 6, 37: 3, 1: 1, 35: 1, 6: 1, 23: 1, 15: 1})
# chekc_feat :  35
# curr_entropy_loss :  0.266764987803
# curr_entropy_loss :  0.263641090028
# curr_entropy_loss :  0.260360870105
# 
# :
# :
# :
# curr_entropy_loss :  0.260360870105
# curr_entropy_loss :  0.263641090028
# entropy_for_featrue :  0.147505113538
# split_feature :  [24, 15, 24, 35, 37, 23, 24, 37, 6, 24, 24, 37, 1, 24, 37, 6]
# count_split_feature
#  Counter({24: 6, 37: 4, 6: 2, 1: 1, 35: 1, 23: 1, 15: 1})

# In[24]:


def accuracy_score(Y_true, Y_predict):
    Y_true = Y_true.astype(int)
    t = .0
    for i in range(np.size(Y_true)):
        if Y_true[i] == Y_predict[i]:
            t += 1
    return t / np.size(Y_predict)


# In[25]:


def pred(tree, x):
    if tree.is_leaf or different_labels(tree.current_results) == 1:
        return np.argmax(tree.current_results)
    else:
        if x[tree.feature] >= tree.value:
            return pred(tree.true_branch, x)
        else:
            return pred(tree.false_branch, x)


# In[26]:


if __name__ == '__main__':
    stats = evaluate_performance()
    print('Accuracy= ', stats[0, 0])

