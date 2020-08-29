import numpy as np
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt


class gbm:
	def __init__(self, x, y, x_split_val=0.1, tree=3):
		self.x = x
		self.y = y
		self.x_split_val = x_split_val
		self.tree = tree


	# xmin과 xmax를 구해서 그 사이 구간을 x_split_val만큼 증가시켜서 splited_x 리스트에 저장하는 함수
	# 이 리스트가 x_threshold로 사용됨
	def split_x(self, x, x_split_val):
		x_sort = np.sort(x, axis=0)
		xmin, xmax = x_sort[0], x_sort[-1]
		splited_x = []
		while xmin < xmax:
			splited_x.append(xmin)
			xmin = xmin + x_split_val
		return splited_x


	# x_threshold 값을 기준으로 나눌 때, 각 x와 y의 값을 리스트로 반환하는 함수
	def split(self, x, y, x_threshold):
		indices_left = [i for i, x_i in enumerate(x) if x_i <= x_threshold]
		indices_right = [i for i, x_i in enumerate(x) if x_i > x_threshold]

		x_left = np.array([x[i] for i in indices_left])
		y_left = np.array([y[i] for i in indices_left])
		x_right = np.array([x[i] for i in indices_right])
		y_right = np.array([y[i] for i in indices_right])
		return x_left, y_left, x_right, y_right


	# split함수에서 받은 x_left, y_left, x_right, y_right로
	# x를 기준으로 나뉜 x의 왼쪽과 x의 오른쪽에 존재하는 y값의 평균을 구하고,
	# 그 평균을 기존의 y에서 빼서 각각의 잔차(residual)를 구한다.
	# --> (매번 tree를 만들기 위해서) 최적의 y split값을 찾기 위한 리스트들을 만든다고 보면 됨.
	def split_y(self, x, y):
		x_thresholds = self.split_x(x, self.x_split_val)
		y_list, left_list, right_list = [], [], []
		for j in range(len(x_thresholds)):
			x_left, y_left, x_right, y_right = self.split(x, y, x_thresholds[j])
			# 			print("x_left, y_left, x_right, y_right:",x_left, y_left, x_right, y_right)
			split_y_left = np.mean(y_left)
			y_left_residual = y_left - split_y_left
			left_list.append(split_y_left)

			split_y_right = np.mean(y_right)
			y_right_residual = y_right - split_y_right
			right_list.append(split_y_right)

			y_list.append(np.append(y_left_residual, y_right_residual))
		return y_list, x_thresholds, left_list, right_list

	# squared loss 함수를 이용해서 가장 최적의 new_y를 찾아가면서 tree를 구축할 수 있는
	# 각각의 new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, error을 반환하기 위한 함수
	def select_residual(self):
		new_x = None
		new_y = self.y
		l_ = None
		r_ = None
		error = np.inf
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
				q_list.append(self.squared_loss(q))
				min_error = min(q_list)
				new_y = selected_y_list[q_list.index(min(q_list))]
				new_x = x_thresholds[q_list.index(min(q_list))]
				l_ = left_list[q_list.index(min(q_list))]
				r_ = right_list[q_list.index(min(q_list))]
				if min_error < error:
					error = min_error
					beststump['s'] = s
		return new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, error

	# loss function for regression - squared loss
	def squared_loss(self, selected_list):
		return sum(0.5 * ((selected_list) ** 2))

	# select_residual함수에서 얻은 최적의 x, x 기준 왼쪽 y, 오른쪽 y의 최적값을 가지고,
	# testdata가 들어왔을 때, y를 예측하는 함수
	def predict(self, testdata):
		new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, error = self.select_residual()

		predicted_val_list = []
		for m in range(len(testdata)):
			residual_sum = []
			for n in range(len(split_y_left_list) - 1):
				if testdata[m] <= new_x_list[n + 1]:
					residual_sum.append(split_y_left_list[n + 1])
				else:
					residual_sum.append(split_y_right_list[n + 1])
			print(testdata[m], "predict value: ", sum(residual_sum))
			predicted_val_list.append(sum(residual_sum))
		return predicted_val_list


# 사용한 데이터
def loadSimpleData():
	train_data = np.array(
		[[1], [1.8], [3], [3.7], [4], [4.2], [4.5], [5], [5.3], [5.4], [5.9], [6.8], [7], [7.5], [7.6], [7.7],
		 [8.1], [8.3], [9], [9.5]])
	train_label = np.array(
		[2.2, 0.7, 0.6, 0.9, 1, 1.4, 1.5, 0.8, -0.7, -0.8, -0.9, 0.4, 0.6, -0.7, -1.0, -1.2, -1.5, 1.6, -1.1, 0.9])
	# train_data = np.array([[1], [1.8], [3], [3.7], [4], [4.2], [4.5], [5], [5.3], [5.4]])
	# train_label = np.array([2.2, 0.7, 0.6, 0.9, 1, 1.4, 1.5, 0.8, -0.7, -0.8])
	test_data = np.array([[1], [3], [3.7], [4.2], [4.5]])
	test_label = [2.2, 0.6, 1, 1.4, 1.5]
	return train_data, train_label, test_data, test_label


train_data, train_label, test_data, test_label = loadSimpleData()

if __name__ == '__main__':
	start = time.time()
	print("start gbm...")
	GBM = gbm(x=train_data, y=train_label, x_split_val=0.1, tree=160) # x값은 0.1씩 움직이면서 160개 tree를 만들어보자!
	new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, error = GBM.select_residual()
	# print("best_tree: ", beststump.get('s'))
	predicted_label = GBM.predict(testdata=test_data) # testdata를 넣어서 예측해보자!
	print("test_label: ", test_label) # 실제 테스트 데이터의 y값
	print("predicted_label: ", predicted_label) # 예측된 값

	mse = mean_squared_error(test_label, predicted_label) # mse값
	print("MSE: %.4f" % mse)
	end = time.time()
	print("time: ", end-start)

	### ##### plot ####### ###
	i = 1
	plt.figure(figsize=(10, 20))
	for e in range(1, 160, 40): #160까지 돌렸을 때, 40번째 tree마다 그래프로 확인해보자!
		plt.subplot(410 + i)

		plt.scatter(train_data, new_y_list[e - 1], s=10, c='black')

		x_split_val = new_x_list[e][0]
		y_left_val = split_y_left_list[e]
		y_right_val = split_y_right_list[e]

		plt.plot([x_split_val,x_split_val],[y_left_val,y_right_val], color='green')
		plt.plot([0, x_split_val], [y_left_val, y_left_val], color='red')
		plt.plot([x_split_val, 10], [y_right_val, y_right_val], color='red')
		plt.grid(True)

		plt.xlim(0, 10)
		plt.ylim(-2.0, 2.5)
		plt.xlabel("x", fontsize=12)
		plt.ylabel("y", fontsize=12)
		plt.annotate(str(round(x_split_val, 3)), xy=(x_split_val, (y_left_val+y_right_val)+0.1),
		             xytext=(x_split_val, (y_left_val+y_right_val)+0.4), arrowprops=dict(arrowstyle="->"), fontsize=10)
		plt.annotate(str(round(y_left_val, 3)), xy=(x_split_val-1, y_left_val+0.5),
		             xytext=(x_split_val-1, y_left_val+0.1), fontsize=10)
		plt.annotate(str(round(y_right_val, 3)), xy=(x_split_val+1, y_right_val-0.5),
		             xytext=(x_split_val+1, y_right_val-0.3), fontsize=10)
		plt.title("Tree"+str(e))
		plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

		i = i + 1
	plt.tight_layout()
	plt.show()