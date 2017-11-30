import numpy as np
import math
import operator
 

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1

    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
  
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def centroid(neighborcoor,k,length):
    centro=[]   
    for x in range(length):
        mean_coor=0
        for y in range(k):
            mean_coor += neighborcoor[y][x]
        centro.append(mean_coor/k)
    return centro


def meanDistance(neighborcoor,testInstance,k):
    meandist = []
    length = len(testInstance)-1
    center = centroid(neighborcoor,k,length)
    print('center:',center)
    dist = euclideanDistance(testInstance,center,length)
    meandist.append(dist)
    return meandist


def main():
	
    # prepare data
    #trainingSet = [[1,2,'-'],[2,2,'-'],[3,2,'-'],[4,2,'-'],[1,1,'-'],[2,1,'-'],[3,1,'-'],[4,1,'-']]
    #testSet = [[2.5,2.5,'-'],[0.7,1,'-']]
    trainingSet = [[1,1,'-'],[2,1,'-'],[3,1,'-'],[1.5,2,'-'],[2.5,2,'-']]
    testSet = [[1.15,1.2,'-'],[2,2.2,'-']]
    
    k = 4

    neighbors_inform=[]
    knn_dist_mean=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        neighbors=np.array(neighbors)
        neighbors_inform.append(neighbors)
        
        neighborcoordinate = np.array(neighbors)[:,0]
        meanDist = meanDistance(neighborcoordinate,testSet[x],k)
        knn_dist_mean.append(meanDist)
    
    print('knn_dist_mean:',knn_dist_mean) 

    #print('neighbors_inform:',neighbors_inform)
    #neighbors_dist=np.array(neighbors_inform)[:,:,1]
    #knn_dist_max=np.max(neighbors_dist,axis=1)
    #knn_dist_avg=np.mean(neighbors_dist,axis=1)
    #print('knn_dist_max:',knn_dist_max)
    #print('knn_dist_avg:',knn_dist_avg)
    

main()