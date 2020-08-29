import math
import numpy as np
import operator
from collections import namedtuple 
import matplotlib.pyplot as plt  


########### Convex Hull의 좌표 구하기 for ConvexHull Distance #############
Point = namedtuple('Point', 'x y')

class ConvexHull(object):  

    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:
            
            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1
 
            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2


            self._hull_points.append(far_point)
            point = far_point


    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')

        # hull points
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()


########### Convex Hull까지의 거리 구하기 for ConvexHull Distance #############

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def getconvexdist(arrtestSet,getConvexHullPoint):

    pnt2lineset=[]
    for i in range(arrtestSet.shape[0]):
        for j in range(getConvexHullPoint.shape[0]-1):
        #pnt2line(pnt, getConvexHullPoint[i], getConvexHullPoint[i+1])
            result=pnt2line(arrtestSet[i], getConvexHullPoint[j], getConvexHullPoint[j+1])
            pnt2lineset.append(result[0])

    pnt2lineset=np.reshape(pnt2lineset,(arrtestSet.shape[0],getConvexHullPoint.shape[0]-1))
    convex_dist=np.min(pnt2lineset,axis=1)
    print('pnt2lineset:',pnt2lineset)

    return convex_dist

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)


########### KNN Information구하기 for KNN Avg Distance #############

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

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


def main():  
    ch = ConvexHull()

#    trainingSet = [[1,2,'s'],[2,2,'s'],[3,2,'s'],[4,2,'s'],[1,1,'s'],[2,1,'s'],[3,1,'s'],[4,1,'s']]
#    testSet = [[2.5,2.5,'a'],[0.7,1,'a']]

    trainingSet = [[1,1,'s'],[3,1,'s'],[2,1,'s'],[1.5,2,'s'],[2.5,2,'s']]
    testSet = [[1.15,1.2,'a'],[2,2.2,'a']]

    arrtrainingSet=np.array(trainingSet)
    arrtestSet=np.array(testSet)

    arrtrainingSet=arrtrainingSet[:,:-1].astype(np.float)
    arrtestSet=arrtestSet[:,:-1].astype(np.float)
  
    for i in range(arrtrainingSet.shape[0]):
        ch.add(Point(arrtrainingSet[i,0], arrtrainingSet[i,1]))

#    for _ in range(6):
#        ch.add(Point(random.randint(-100, 100), random.randint(-100, 100)))

    #print("Points on hull:", ch.get_hull_points())
    getConvexHullPoint=np.array(ch.get_hull_points())
    ch.display()

    print('ConvexHullPoint:',getConvexHullPoint)

    convex_dist=getconvexdist(arrtestSet,getConvexHullPoint)

    print('convex_dist:',convex_dist)
        
    k = 5
    neighbors_inform=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        neighbors_inform.append(neighbors)
        
    neighbors_dist=np.array(neighbors_inform)[:,:,1]
    knn_dist_avg=np.mean(neighbors_dist,axis=1)
    
    #print('neighbors_dist:',neighbors_dist)
    print('knn_dist_avg:',knn_dist_avg)
    
    knn_dist_hybrid=knn_dist_avg*2/(1+np.exp(-convex_dist))
    print('knn_dist_hybrid:',knn_dist_hybrid)
    
    
main()