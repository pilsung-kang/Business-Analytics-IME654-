import numpy as np
import random
import csv
import matplotlib.pyplot as plt



#파일로 데이터 입력(Column Name없이), 파일명 : ex_kmc.csv
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
  
           
def kmeans(X,k,maxIt):
    numPoints,numDim=X.shape

    dataSet=np.zeros((numPoints,numDim+1))
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k),:]
    centroids[:,-1]=range(1,k+1)
    #print("centroids:",centroids)

    iterations=0;
    oldCentroids=None
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        oldCentroids=np.copy(centroids)
        iterations+=1
        updateLabels(dataSet, centroids)
        centroids=getCentroids(dataSet, k)
    return dataSet, centroids


def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)

def updateLabels(dataSet,centroids):
    numPoints,numDim=dataSet.shape
    for i in range(numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1], centroids)

def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
            #print("label:"+str(label))
    return label


def getCentroids(dataSet,k):       
    result=np.zeros((k,dataSet.shape[1]))
    #print("result:",result)
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        #print("cluster:",oneCluster)
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    #print("result:",result)
    return result


def getNoveltyScore(dataSetRow,centroids,k):
    minDist=[]  
    maxDist=[]
    for i in range(k):
        Dist=np.linalg.norm(dataSetRow[:,:-1]-centroids[i,:-1],axis=1)   
        minDist.append(Dist)     
    
    NoveltyScore=np.min(minDist,axis=0)
    min_index=np.argmin(minDist,axis=0)
    
    for i in range(k):
        clusterDist=NoveltyScore[np.where(min_index==i)]   
        maxDist.append(np.max(clusterDist))
    
#    a=centroids[min_index,-1]
#    print(a)
    #print(NoveltyScore)
    return NoveltyScore,maxDist


def getModNoveltyScore(kmeans_result,centroid_result,maxDist,k):
    minDist=[]
    countcluster=[]    
    for i in range(k):
        Dist=np.linalg.norm(kmeans_result[:,:-1]-centroid_result[i,:-1],axis=1)
        minDist.append(Dist)
        count=np.sum(kmeans_result[:,-1]==i+1)
        countcluster.append(count)           

    NoveltyScore=np.min(minDist,axis=0)
    weight=(countcluster/np.min(countcluster))/maxDist
    scale_weight=weight/np.min(weight)
    mod_NoveltyScore = NoveltyScore*np.repeat(scale_weight,countcluster,axis=0)

#    print(np.repeat(scale_weight,countcluster,axis=0))
#    print(countcluster)
#    print(np.min(countcluster))
#    print(scale_weight)
#    print(minDist)
#    print(MaxDist)
#    print(NoveltyScore)
   
    return mod_NoveltyScore


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 1
    random.seed(100)
    loadDataset('C:/Users/myunghoon/.spyder-py3/ex_kmc2.csv', split, trainingSet, testSet)
    trainX=np.array(trainingSet)
    
    #print('Train set: ' ,repr(len(trainingSet)))
    #print('Test set: ' + repr(len(testSet)))
    #print('Train set: ' ,trainingSet[0:10])
    #a=trainX[:,:-1].astype(np.float)
    #plt.scatter(a[:,0], a[:,1]);
    
    # k=군집개수, max_iter=반복 회수 제한(Hyper parameter)   
    k=2
    max_iter=100
    final_result=kmeans(trainX[:,:-1],k,max_iter)

    kmeans_result=final_result[0]
    centroid_result=final_result[1]
    
    NoveltyScore_result=getNoveltyScore(kmeans_result,centroid_result,k)  
    Score=NoveltyScore_result[0]
    maxDist=NoveltyScore_result[1]
    
    print('final centroid:',centroid_result)
    print('maxDist:',maxDist)
    print('Novelty Score:',Score)


    #Outlier제거
    outlier_percentage=0.98
    Distsorting=np.argsort(Score)
    outlier_point=int(Distsorting.shape[0]*outlier_percentage)    
    cutoff_index=Distsorting[0:outlier_point]
    trainX=trainX[cutoff_index]
    
    #두번째 K-Means (새로운 Centroid와 군집의 크기(MaxDist)를 구함)
    final_result_second=kmeans(trainX[:,:-1],k,max_iter)
    kmeans_result_second=final_result_second[0]
    centroid_result_second=final_result_second[1]
    
    NoveltyScore_result_second=getNoveltyScore(kmeans_result_second,centroid_result_second,k)  
#    Score_second=NoveltyScore_result_second[0]
    maxDist_second=NoveltyScore_result_second[1]

    #Relative Distance기반의 Novelty Score    
    RelativeScore=getModNoveltyScore(kmeans_result,centroid_result_second,maxDist_second,k)

    print('final centroid_second:',centroid_result_second)
    print('maxDist_second:',maxDist_second)
    print('Relative Novelty Score:',RelativeScore)    
     
    x=kmeans_result[:,0]
    y=kmeans_result[:,1]
    colors=kmeans_result[:,2]
    
    plt.figure(figsize=(7, 3), dpi=80)
    plt.scatter(x, y, s=(RelativeScore**2)*5, c=colors,alpha=0.5);
    plt.xticks(np.arange(-1,8,2))
    plt.yticks(np.arange(-1.5,1.6,0.5))
    
main()





