from sklearn import (datasets,random_projection)
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from isomap import isomap
from mpl_toolkits.mplot3d import Axes3D

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30
X_tmp = X[:10,:]
#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

    # Plot images of the digits
n_img_per_row = 30
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')


# Random 2D projection using a random unitary matrix
print("Computing random projection")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits")



# Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time()
embedding = isomap(input=X, n_neighbors=n_neighbors, n_components=2, n_jobs=4)
print("Done.")
plot_embedding(embedding,"Isomap projection of the digits (time %.2fs)" %(time() - t0))


embedding_three_dim = isomap(input=X, n_neighbors=n_neighbors, n_components=3, n_jobs=4)
# draw 32 plots
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(embedding_three_dim[:, 0], embedding_three_dim[:, 1], embedding_three_dim[:, 2], c=y)
