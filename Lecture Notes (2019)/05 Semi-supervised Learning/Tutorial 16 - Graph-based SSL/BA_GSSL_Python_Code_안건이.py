
# coding: utf-8

# In[1]:

import os
import numpy as np
import numpy.linalg as lin
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse 
from scipy.sparse.linalg import inv
from scipy.spatial import distance


# In[2]:

os.getcwd()


# In[25]:

datafile = "../GSSL_DATA/Data1.csv"
data = pd.read_csv(datafile)
Testdata = pd.read_csv(datafile)


# In[26]:

data


# In[4]:

#Class를 나눠 주는 작업과 각 Class를의 index에 대한 정보 저장하기
class1_idx = (data['V3'] == '1')
class2_idx = (data['V3'] == '2')
labeled_idx = (class1_idx | class2_idx)
unlabeled_idx = (labeled_idx != True)


# In[27]:

class1_idx


# In[6]:

#String 정보를 숫자로 바꿔 주기
num_samples = data.shape[0]
y = np.zeros(shape=(num_samples))
y[class1_idx] = 1
y[class2_idx] = 2
y[unlabeled_idx] = 0 
data['V3'] = y


# In[7]:

#행렬로 변환
lenght = len(y)
Yl = np.full((lenght,1),0)
Yl[class1_idx] = 1
Yl[class2_idx] = 2
labeled_lenght = len(y[labeled_idx])


# In[8]:

# RBF kernal 함수에 사용할 Distance Matrix 만들기
euc = distance.cdist(data.iloc[:, :2], data.iloc[:, :2], 'sqeuclidean')


# In[9]:

# e_radius 함수 만들기
def e_radius(euc, epsilon):
    if epsilon <= 0:
        print('Use epsilon >= 0')
        return None
    e_distance = np.where(euc < epsilon, euc, np.inf)
    return e_distance


# In[10]:

# e_radius 함수를 가지고 w_matrix 만들기
def RBF_Weight(euc, epsilon, gamma):
    euc = e_radius(euc, epsilon)
    
    #RBF
    w_matrix = np.exp(-euc*gamma) #시그마 제곱 대신 gamma를 사용한다(시그마 제곱의 역수).
    np.fill_diagonal(w_matrix,0)
    return w_matrix

W = RBF_Weight(euc, epsilon = 1, gamma = 20)


# In[11]:

W


# In[12]:

# Diagonal Dgree Matrix 만들기
rowsum = W.sum(axis=1)
D = sparse.diags(rowsum)


# In[13]:

# Laplacian Matrix 만들기
L = D - W


# In[14]:

# Laplacian Matrix 중에서 필요한 Subset Matrix 추출
Luu = L[labeled_lenght:,labeled_lenght:]
Lul = L[labeled_lenght:,:labeled_lenght]


# In[15]:

Luu.shape


# In[16]:

Lul.shape


# In[30]:

# Unlabeled data에 Label 부여하기
Fu = -lin.inv(Luu)*Lul*Yl[labeled_idx]


# In[31]:

Fu


# In[28]:

# Mincut이 아니기 때문에 1.5보다 큰것은 2로 1.5보다 작은 것은 1로 가게 하기
Fu_lenght = len(Fu)
for i in range(Fu_lenght):
    if Fu[i,0] >= 1.5:
        Fu[i,0] = 2
    else:
        Fu[i,0] = 1


# In[29]:

Fu


# In[19]:

# Total Data에 새로 부여된 labeled data를 추가해주기
Total_y_lenght = len(y[class1_idx]) + len(y[class2_idx]) + len(y[unlabeled_idx])
Total_y =  np.full((Total_y_lenght,1),0)
Total_y[class1_idx] = 1
Total_y[class2_idx] = 2
Total_y[unlabeled_idx] = Fu 


# In[20]:

Testdata['V3'] = Total_y
Testdata


# In[21]:

plt.scatter(data['V1'],data['V2'],c=data['V3'])
plt.show()


# In[22]:

plt.scatter(Testdata['V1'],Testdata['V2'],c=Testdata['V3'])
plt.show()


# In[43]:

I = sparse.eye(L.shape[0])


# In[44]:

I


# In[68]:

Lam = 2.0


# In[69]:

YL = np.full((len(y),1),0)


# In[70]:

YL[class1_idx] = 1
YL[class2_idx] = 2
YL[unlabeled_idx] = 1


# In[71]:

L.shape


# In[72]:

Fu_ = lin.inv(I + Lam*L)*YL


# In[73]:

Fu_


# In[75]:

Fu__lenght = len(Fu_)
for i in range(Fu__lenght):
    if Fu_[i,0] >= 1.4:
        Fu_[i,0] = 2
    else:
        Fu_[i,0] = 1


# In[76]:

Total_y_lenght_ = len(y[class1_idx]) + len(y[class2_idx]) + len(y[unlabeled_idx])
Total_y_ =  np.full((Total_y_lenght_,1),0)
Total_y_ = Fu_ 


# In[77]:

Testdata['V3'] = Total_y_
Testdata


# In[78]:

plt.scatter(Testdata['V1'],Testdata['V2'],c=Testdata['V3'])
plt.show()

