import numpy as np
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt

class gbm:
	def __init__(self, x, y, x_split_val=0.1, tree=160, error=0.1):
		self.x = x
		self.y = y
		self.x_split_val = x_split_val
		self.tree = tree
		self.error = error #앞 코드에서 추가된 부분

	def split_x(self, x, x_split_val):
		x_sort = np.sort(x, axis=0)
		xmin, xmax = x_sort[0], x_sort[-1]
		splited_x = []
		while xmin < xmax :
			splited_x.append(xmin)
			xmin = xmin + x_split_val
		return splited_x


	def split(self, x, y, x_threshold):
		indices_left = [i for i, x_i in enumerate(x) if x_i <= x_threshold]
		indices_right = [i for i, x_i in enumerate(x) if x_i > x_threshold]

		x_left = np.array([x[i] for i in indices_left])
		y_left = np.array([y[i] for i in indices_left])
		x_right = np.array([x[i] for i in indices_right])
		y_right = np.array([y[i] for i in indices_right])
		return x_left, y_left, x_right, y_right


	def split_y(self, x, y):
		x_thresholds = self.split_x(x, self.x_split_val)
		y_list, left_list, right_list = [], [], []
		for j in range(len(x_thresholds)):
			x_left, y_left, x_right, y_right = self.split(x, y, x_thresholds[j])
			split_y_left = np.mean(y_left)
			y_left_residual = y_left - split_y_left
			left_list.append(split_y_left)

			split_y_right = np.mean(y_right)
			y_right_residual = y_right - split_y_right
			right_list.append(split_y_right)
			y_list.append(np.append(y_left_residual, y_right_residual))
		return y_list, x_thresholds, left_list, right_list


	def select_residual(self):
		new_x = None
		new_y = self.y
		l_ = None
		r_ = None
		min_error = np.inf
		new_x_list, new_y_list, split_y_left_list, split_y_right_list = [], [], [], []
		beststump = {}

		for s in range(self.tree):
			selected_y_list, x_thresholds, left_list, right_list = self.split_y(self.x, new_y)
			q_list = []

			new_x_list.append(new_x)
			new_y_list.append(new_y)

			split_y_left_list.append(l_)
			split_y_right_list.append(r_)

			for u in range(len(selected_y_list)):
				q = selected_y_list[u]
				q_list.append(0.5*sum(q**2))
				min_error = min(q_list)
				new_y = selected_y_list[q_list.index(min(q_list))]
				new_x = x_thresholds[q_list.index(min(q_list))]
				l_ = left_list[q_list.index(min(q_list))]
				r_ = right_list[q_list.index(min(q_list))]
			### 앞 코드와 다른 부분 ###
			if (min_error < self.error):
				beststump['s'] = s
				break
			else:
				continue
			### ################## ###
		return new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, min_error


	def predict(self, testdata):
		new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, min_error = self.select_residual()

		predicted_val_list = []
		for m in range(len(testdata)):
			residual_sum = []
			for n in range(len(split_y_left_list)-1):
				if testdata[m] <= new_x_list[n+1]:
					residual_sum.append(split_y_left_list[n+1])
				else:
					residual_sum.append(split_y_right_list[n+1])
			print(testdata[m], "predict value :", sum(residual_sum))
			predicted_val_list.append(sum(residual_sum))
		return predicted_val_list


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


if __name__ == '__main__':
	start = time.time()
	print("start gbm...")
	GBM = gbm(x=train_data, y=train_label, x_split_val=0.1, tree=160, error=0.1) #앞 코드와 달리 error값 추가 - error:0.1
	# GBM = gbm(x=train_data, y=train_label, x_split_val=0.1, tree=160, error=0.01) #앞 코드와 달리 error값 추가 - error:00.1
	# GBM = gbm(x=train_data, y=train_label, x_split_val=0.1, tree=160, error=0.001) #앞 코드와 달리 error값 추가 - error:000.1

	new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, min_error = GBM.select_residual()
	print("best_tree:", beststump.get('s'))

	predicted_label = GBM.predict(testdata=test_data)
	print("test_label: ", test_label)
	print("predicted_label: ", predicted_label)

	mse = mean_squared_error(test_label, predicted_label)
	print("MSE: %.4f" % mse)
	end = time.time()
	print("time: ", end-start)
