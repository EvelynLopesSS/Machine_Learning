import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
x = np.array([[1,2],
              [1.3,14],
              [4,7],
              [45,11],
              [30,20],
              [40,30],
              [30,30],
              [19,40],
              [9,10],
             [1.0,6],
             [8,8],
             [5,8]])

plt.scatter(x[:,0], x[:,1], s=150)
plt.show()
colors = 10*['g','r','c','b','k']
class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []

            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))
            prev_centroids = dict(centroids)
            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        pass


clf = Mean_Shift()
clf.fit(x)

centroids = clf.centroids

plt.scatter(x[:, 0], x[:, 1], s=150)
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c in centroids:
    ax.scatter(centroids[c][0], centroids[c][1], c, color='k', marker='*', s=150)

ax.scatter(x[:, 0], x[:, 1], np.zeros(len(x)), s=150)
plt.show()