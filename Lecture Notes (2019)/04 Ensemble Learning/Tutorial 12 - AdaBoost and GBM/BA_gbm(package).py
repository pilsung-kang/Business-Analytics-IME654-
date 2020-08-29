import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def loadSimpleData():
	train_data = np.array([[1], [1.8], [3], [3.7], [4], [4.2], [4.5], [5], [5.3], [5.4], [5.9], [6.8], [7], [7.5], [7.6], [7.7],
	                 [8.1], [8.3], [9], [9.5]])
	train_label = np.array([2.2, 0.7, 0.6, 0.9, 1, 1.4, 1.5, 0.8, -0.7, -0.8, -0.9, 0.4, 0.6, -0.7, -1.0, -1.2, -1.5, 1.6, -1.1, 0.9])
	# train_data = np.array([[1], [1.8], [3], [3.7], [4], [4.2], [4.5], [5], [5.3], [5.4]])
	# train_label = np.array([2.2, 0.7, 0.6, 0.9, 1, 1.4, 1.5, 0.8, -0.7, -0.8])
	test_data = np.array([[1], [3], [3.7], [4.2], [4.5]])
	test_label = [2.2, 0.6, 1, 1.4, 1.5]
	return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = loadSimpleData()
params = {'n_estimators': 160, 'max_depth': 1,
          'learning_rate': 1, 'loss': 'ls'}

regr = ensemble.GradientBoostingRegressor(**params)
regr.fit(train_data, train_label)
mse = mean_squared_error(test_label, regr.predict(test_data))
print("MSE: %.4f" % mse)

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(regr.staged_predict(test_data)):
	test_score[i] = regr.loss_(test_label, y_pred)
print("test_label: ", test_label)
print("i: {}, y_pred: {} ".format(i, y_pred))