import numpy as np
import random
import csv
import matplotlib.pyplot as plt


def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def pca(X,n): 

    #1st Step : subtract mean 
    avg = np.mean(X,axis=0)
    avg = np.tile(avg,(X.shape[0],1)) 
    X -= avg; 
    #print(avg)

    #2nd Step : covariance matrix 
    C = np.dot(X.transpose(),X)/(X.shape[0]-1)

    #3rd Step : Eigen Value, Eigen Vector 
    eig_values,eig_vecs = np.linalg.eig(C)

    #4rd Step : Select n개의 PC 
    idx = np.argsort(eig_values)[-n:][::-1]
    eig_values = eig_values[idx] 
    eig_vecs = eig_vecs[:,idx] 
    
    #print(eig_values.argsort())
    #print(eig_values.argsort()[-n:])
    #print(idx)
    #print(eig_values)
    #print(eig_vecs)
 
    #5th Step : new coordinate in new space 
    Y = np.dot(X,eig_vecs) 

    #6th Step : reconstruction 
    rec=np.dot(eig_vecs,Y.transpose())
    Score=np.linalg.norm(X.transpose()-rec,axis=0)     
    #print(rec)
    #print(Score)
    
    return (X.transpose(), rec, Score.transpose(), eig_vecs, eig_values) 


def main():

	# prepare data
    trainingSet=[]
    testSet=[]
    split = 0.8
    random.seed(100)
    loadDataset('ex_pca5.csv', split, trainingSet, testSet)
    #print('Train set: ' + repr(len(trainingSet)))
    #print('Test set: ' + repr(len(testSet)))

    # n=PC개수 (Hyper parameter)  
    n=1
    trainX=np.array(trainingSet)
    pca_result=pca(trainX[:,:-1].astype(np.float),n)

   # print('pca result:',pca_result)

    x=pca_result[0][0]
    y=pca_result[0][1]
    Score=pca_result[2]*100

    print('Eigen Value : ', pca_result[4])
    print('Eigen Vector : ', pca_result[3])    
    print('Data X : ', np.transpose(pca_result[0])[:10])
    print('Reconstruction : ', np.transpose(pca_result[1])[:10])
    print('Novelty Score : ', np.transpose(pca_result[2])[:10])
        
    x_rec=pca_result[1][0]
    y_rec=pca_result[1][1]
    
    plt.figure(figsize=(6, 6), dpi=80)
    plt.scatter(x,y,s=Score);
    plt.scatter(x_rec,y_rec,s=20);
    plt.xticks(np.arange(-3,3,0.5))
    plt.yticks(np.arange(-3,3,0.5))
         
main()