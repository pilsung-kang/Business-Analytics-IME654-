# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:46:21 2017

@author: 준헌
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Cotraining을 구성하는데 필요한 패키지
import random
import numpy as np

# Cotraining에 적용 시킬 알고리즘
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# 부가 기능을 위한 패키지
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class CoTraining:

    def __init__(self, clf1, clf2, p, n, k, u):
        self.clf1 = clf1    # 첫 번째 Classifier
        self.clf2 = clf2    # 두 번째 Classifier
        self.p = p   # U'에서 1회 동안 Class 1로 정의할 instance 개수
        self.n = n   # U'에서 1회 동안 Class 0으로 정의할 instance 개수
        self.k = k   # U'를 업데이트 하면서 Label을 다는 과정의 반복 수
        self.u = u   # U'의 크기
        
        # hyper parameter가 0보다 작은 경우 Assert error
        assert(self.p > 0 and self.n > 0 and self.k > 0 and self.u > 0)
        random.seed()


    def fit(self, X1, X2, y):  #Co-training을 수행 함수

		# Class의 정보가 -1인 경우 Unlabeled Data로 정의
        U = [i for i, y_i in enumerate(y) if y_i == -1]

		# U'를 랜덤하게 추출하기 위해서 Unlabeled data를 Shuffle
        random.shuffle(U)

		# parameter u의 크기로 U'를 정의
        U_ = U[-min(len(U), self.u):]

		# Unlabeled data에서 U'크기 만큼을 제외
        U = U[:-len(U_)]

		# Class의 정보가 -1이 아닌 경우 labeled Data로 정의
        L = [i for i, y_i in enumerate(y) if y_i != -1]

        it = 0 

		# k번 만큼 Unlabeled data를 Labeled data로 변
        while it != self.k and U:
            it += 1
            
            # Labeled data를 통해 두 개의 Classifier를 학습
            self.clf1.fit(X1[L], y[L])
            self.clf2.fit(X2[L], y[L])
            
            # Unlabeled data를 학습된 Classifier로 Class 예측
            y1 = self.clf1.predict(X1[U_])
            y2 = self.clf2.predict(X2[U_])
            # Unlabeled data를 학습된 Classifier로 probability 예측
            y1_proba = self.clf1.predict_proba(X1[U_])
            y2_proba = self.clf2.predict_proba(X2[U_]) 
            
            # 두 Classifier의 예측 Class가 다른 instance는 확률을 -1로 변경
            # 이 경우는 각 Classifier의 예측확률이 높아도 배제하기 위함
            for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
                if y1_i != y2_i:
                    y1_proba[i,:] = -1
                    y2_proba[i,:] = -1
            
            n, p = [], []
            y1_p_flag = 0
            y1_n_flag = 0
            y2_p_flag = 0
            y2_n_flag = 0
            # 각 Classifier의 예측 Class가 동일하면서 예측확률이 높은 순으로 
            # Label data로 변경할 Unlabeled data를 선정
            while True:
                # 확률이 높은 instance의 확률 값 저장
                y1_n_proba = y1_proba[:,0].max()
                y1_p_proba = y1_proba[:,1].max()
                y2_n_proba = y2_proba[:,0].max()
                y2_p_proba = y2_proba[:,1].max()
                
                # 확률이 0.6미만이면 Break
                if y1_n_proba < 0.6:
                    break
                # 2p + 2n 만큼의 Unlabeled data 선정되면 Break
                if len(p) == 2 * self.p and len(n) == 2 * self.n:
                    break
                
                # 각 classifier 별 n,p 개수만큼 확률이 최대인 instance의 인덱스 추출 
                for i, (y1_n_i, y1_p_i, y2_n_i, y2_p_i) in enumerate(zip(y1_proba[:,0],y1_proba[:,1], y2_proba[:,0], y2_proba[:,1])):
                    if y1_n_i == y1_n_proba and y1_n_flag < self.n:
                        n.append(i)
                        y1_n_flag += 1
                        y1_proba[i,:] = -1
                        y2_proba[i,:] = -1
                        continue
                    if y1_p_i == y1_p_proba and y1_p_flag < self.p:
                        p.append(i)
                        y1_p_flag += 1
                        y1_proba[i,:] = -1
                        y2_proba[i,:] = -1
                        continue
                    if y2_n_i == y2_n_proba and y2_n_flag < self.n:
                        n.append(i)
                        y2_n_flag += 1
                        y1_proba[i,:] = -1
                        y2_proba[i,:] = -1
                        continue
                    if y2_p_i == y2_p_proba and y2_p_flag < self.p:
                        p.append(i)    
                        y2_p_flag += 1
                        y1_proba[i,:] = -1
                        y2_proba[i,:] = -1
                        continue       
            # 추출된 instance의 index에 Labeling        
            y[[U_[x] for x in p]] = 1
            y[[U_[x] for x in n]] = 0
            # Labeling된 instance를 Label data로 추가
            L.extend([U_[x] for x in p])
            L.extend([U_[x] for x in n])
            
            # 추출된 instance의 index를 U'에서 제거
            remove = p + n
            remove.sort()
            remove.reverse()
            for i in remove: U_.pop(i)
            # U'에서 빠진만큼 Unlabeled data에서 채워 넣기
            add_cnt = 0 
            num_to_add = len(remove)
            while add_cnt != num_to_add and U:
                add_cnt += 1
                U_.append(U.pop())

        # 최종적으로 추가완료 된 Labeled data로 각 Classifier를 학습
        self.clf1.fit(X1[L], y[L])
        self.clf2.fit(X2[L], y[L])
	
    # 두개의 Classifier의 Class예측 확률의 합쳐서 최종 Class 예측
    def predict(self, X1, X2):

        y1 = self.clf1.predict(X1)
        y2 = self.clf2.predict(X2)

        # 두 Classifier가 동일한 Class시 예측하면 해당 Class로 판정
        # 그렇지 않은 경우, 확률의 합의 비교하여 높은 Class로 판정
        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            if y1_i == y2_i:
                y_pred[i] = y1_i
            else:
                y1_probs = self.clf1.predict_proba([X1[i]])
                y2_probs = self.clf2.predict_proba([X2[i]])
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = sum_y_probs.index(max_sum_prob)
        return y_pred
    
    # 판정을 첫번째 Classifier를 통해서만 예측
    def predict_clf1(self, X1, X2):

        y_pred = self.clf1.predict(X1)

        return y_pred





if __name__ == '__main__':   
    accuracy = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for iteration in range(1,2,1):
        
        #평가하기 위한 데이터 생성
        N_SAMPLES = 5000    # 총 instance 개수
        N_FEATURES = 100    # 총 feature개수
        N_REDUNDANT = 50    # feature간 Correlation를 가지는 feature 개수
        Lable_Percent = 5   # 총 instance중에 Labeled data를 백분율을 설정 (1~99)
        # Data 생성
        X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_redundant=N_REDUNDANT)
        
        #Labeled, Unlabeled data를 설정
        y[:N_SAMPLES*(100-2*Lable_Percent)//100] = -1
        X_test = X[-N_SAMPLES*Lable_Percent//100:]
        y_test = y[-N_SAMPLES*Lable_Percent//100:]
        X_labeled = X[N_SAMPLES*(100-2*Lable_Percent)//100:-N_SAMPLES*Lable_Percent//100]
        y_labeled = y[N_SAMPLES*(100-2*Lable_Percent)//100:-N_SAMPLES*Lable_Percent//100]
        y = y[:-N_SAMPLES*Lable_Percent//100]
        X = X[:-N_SAMPLES*Lable_Percent//100]
        
        # Feature를 반반으로 나눔
        X1 = X[:,:N_FEATURES // 2]
        X2 = X[:, N_FEATURES // 2:]
        

        # Labeled data로만 각 Classifier로 성능 평가
        print ('Logistic')
        base_lr = LogisticRegression()
        base_lr.fit(X_labeled, y_labeled)
        y_pred = base_lr.predict(X_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[0].append('%0.3f'% accuracy_score(y_test, y_pred))
        
        print ('Naive Bayes Classifier')
        base_nb = GaussianNB()
        base_nb.fit(X_labeled, y_labeled)
        y_pred = base_nb.predict(X_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[1].append('%0.3f'% accuracy_score(y_test, y_pred))
        
        print ('Random Forest')
        base_rf = RandomForestClassifier(n_estimators=100)
        base_rf.fit(X_labeled, y_labeled)
        y_pred = base_rf.predict(X_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[2].append('%0.3f'% accuracy_score(y_test, y_pred))   


        # Cotrining을 이용하여 Classifier 조합별 성능 평가    
        print ('Random Forest-Logistic Regression CoTraining')
        lg_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), LogisticRegression(), p=5, n=5, k=50, u=50)
        lg_co_clf.fit(X1, X2, y)
        y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[5].append('%0.3f'% accuracy_score(y_test, y_pred))
        y_pred = lg_co_clf.predict_clf1(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[6].append('%0.3f'% accuracy_score(y_test, y_pred))
        
        print ('Random Forest-Naive Bayes Classifier CoTraining')
        lg_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), LogisticRegression(), p=5, n=5, k=50, u=50)
        lg_co_clf.fit(X1, X2, y)
        y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[7].append('%0.3f'% accuracy_score(y_test, y_pred))
        y_pred = lg_co_clf.predict_clf1(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[8].append('%0.3f'% accuracy_score(y_test, y_pred))
        
        print ('Random Forest-Random Forest CoTraining')
        lg_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=100), p=5, n=5, k=10, u=200)
        lg_co_clf.fit(X1, X2, y)
        y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[3].append('%0.3f'% accuracy_score(y_test, y_pred))
        y_pred = lg_co_clf.predict_clf1(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[4].append('%0.3f'% accuracy_score(y_test, y_pred))

        
        # Feature를 50:50로 나누는 것이 아니라 
        # PCA를 통해 Feature 수만큼 생성하여 Cotrining을 수행한 성능 평가
        from sklearn import decomposition
        from sklearn.preprocessing import normalize
        
        pca = decomposition.PCA(n_components=N_FEATURES)
        pca.fit(normalize(X))
        X_PCA = pca.transform(normalize(X))
        pca = decomposition.PCA(n_components=N_FEATURES)
        pca.fit(normalize(X_test))
        X_PCA_test = pca.transform(normalize(X_test))
        
     
        print ('Random Forest-Random Forest CoTraining with PCA')
        lg_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), RandomForestClassifier(n_estimators=100), p=5, n=5, k=10, u=200)
        lg_co_clf.fit(X, X_PCA, y)
        y_pred = lg_co_clf.predict(X_test, X_PCA_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[9].append('%0.3f'% accuracy_score(y_test, y_pred))
        y_pred = lg_co_clf.predict_clf1(X_test, X_PCA_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[10].append('%0.3f'% accuracy_score(y_test, y_pred))
    	
        print ('Random Forest-Logistic Regression CoTraining with PCA')
        lg_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), LogisticRegression(), p=5, n=5, k=50, u=50)
        lg_co_clf.fit(X, X_PCA, y)
        y_pred = lg_co_clf.predict(X_test, X_PCA_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[11].append('%0.3f'% accuracy_score(y_test, y_pred))
        y_pred = lg_co_clf.predict_clf1(X_test, X_PCA_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[12].append('%0.3f'% accuracy_score(y_test, y_pred))
    
        print ('Random Forest-Naive Bayes CoTraining with PCA')    
        lg_co_clf = CoTraining(RandomForestClassifier(n_estimators=100), GaussianNB(), p=10, n=10, k=1, u=100)
        lg_co_clf.fit(X, X_PCA, y)
        y_pred = lg_co_clf.predict(X_test, X_PCA_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[13].append('%0.3f'% accuracy_score(y_test, y_pred))
        y_pred = lg_co_clf.predict_clf1(X_test, X_PCA_test)
        print (classification_report(y_test, y_pred, digits=3))
        accuracy[14].append('%0.3f'% accuracy_score(y_test, y_pred))
    
    
    print( accuracy)