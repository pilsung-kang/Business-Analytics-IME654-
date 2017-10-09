from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import KernelPCA
from dijkstra import Graph, dijkstra
import numpy as np
import pickle



def isomap(input, n_neighbors, n_components, n_jobs):
    #
    # nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,algorithm='ball_tree',n_jobs=n_jobs)
    # nbrs_.fit(input)
    #
    # kng = kneighbors_graph(X=nbrs_, n_neighbors=n_neighbors, mode='distance')
    # # Build Graph
    # G = Graph()
    # for i in range(len(input)):
    #     G.add_node(i)
    #
    # # Set weight to edges
    # for i in range(len(input)):
    #     for j in range(i+1,len(input)):
    #         if kng.toarray()[i][j] != 0:
    #             G.add_edge(i, j, kng.toarray()[i][j])
    #     print(i)
    #
    # '''
    # #pickle.dump(G, open("./graph_new.p", "wb"))
    # G= pickle.load(open("./graph_new.p","rb"))
    # '''
    #
    # distance_matrix_modified = np.zeros((len(input),len(input)))
    #
    # for i in range(len(input)):
    #     distance_matrix_modified[i][i] = 0
    #     print(i, "/", len(input), "'th Job is started")
    #     dist = dijkstra(G,i)
    #     for j in range(i+1, len(input)):
    #         if j % 100 == 0 :
    #             print(i, "'th Job is still working now", j, "/", len(input))
    #         distance_matrix_modified[i][j] = dist.get(j)
    #         distance_matrix_modified[j][i] = dist.get(j)
    #     print(i, "/", len(input), "'th Job is done")





    distance_matrix = pickle.load(open('./isomap_distance_matrix.p', 'rb'))



    kernel_pca_ = KernelPCA(n_components=n_components,
                                 kernel="precomputed",
                                 eigen_solver='arpack', max_iter=None,
                                 n_jobs=n_jobs)

    Z = distance_matrix ** 2
    Z *= -0.5

    embedding = kernel_pca_.fit_transform(Z)

    return(embedding)

