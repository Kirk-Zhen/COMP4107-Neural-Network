import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from minisom import MiniSom


# https://github.com/JustGlowing/minisom/blob/master/examples/HandwrittenDigits.ipynb
def plot_som(som, data, num, title):
    plt.figure(figsize=(6, 6))
    wmap = {}
    im = 0
    for x, t in zip(data, num):  # scatterplot
        w = som.winner(x)
        wmap[w] = im
        plt.text(w[0] + .5, w[1] + .5, str(int(t)),
                 color=plt.cm.PuOr(t / 5), fontdict={'weight': 'bold', 'size': 11})
        im = im + 1
    plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])

    plt.title(title)
    # plt.savefig('resulting_images/som_digts.png')
    plt.show()

# Data processing
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

combine = np.concatenate((np.concatenate((trX, trY.reshape(-1,1)), axis=1),
                          np.concatenate((teX, teY.reshape(-1,1)), axis=1)), axis=0)

train_set = np.array([row for row in combine if row[-1] == 1 or row[-1] == 5])
np.random.shuffle(train_set)
train_X, train_y = train_set[:,:-1], train_set[:,-1]


# Plot SOM
som = MiniSom(30, 30, 784, sigma=4, learning_rate=0.5, neighborhood_function='triangle')
plot_som(som, train_X, train_y, "SOM Before Training")
som.train_random(train_X, 5000)
plot_som(som, train_X, train_y, "SOM After Training")


# Plot KMeans
num_centroids = 12
pca = PCA(n_components=2)
scaled_X = pca.fit_transform(train_X)
# scale the data between 0 and 1
scaled_X[:,0] = (scaled_X[:,0]-min(scaled_X[:,0]))/(max(scaled_X[:,0])-min(scaled_X[:,0]))
scaled_X[:,1] = (scaled_X[:,1]-min(scaled_X[:,1]))/(max(scaled_X[:,1])-min(scaled_X[:,1]))
# run K-means using the processed data
km = KMeans(init='k-means++', n_clusters=num_centroids, n_init=10)
km.fit(scaled_X)

scalar = np.linspace(-1, 2, 3000, endpoint=True)
xx, yy = np.meshgrid(scalar, scalar, sparse=False)

temp = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
Z = km.predict(temp)
Z = Z.reshape(xx.shape)

# https://matplotlib.org/3.2.0/gallery/images_contours_and_fields/image_masked.html#sphx-glr-gallery-images-contours-and-fields-image-masked-py
plt.figure(figsize=(8,6))
# plot the cluster area
plt.imshow(Z, interpolation='nearest',
           aspect='auto', origin='lower',
           cmap = plt.cm.coolwarm,
           extent=[-1, 2, -1, 2])

# plot the training samples as dots
for lab in [1,5]:
    points = np.array([scaled_X[i] for i in range(train_y.shape[0]) if train_y[i] == lab])
    color = 'orange' if lab == 1 else 'navy'
    plt.plot(points[:, 0], points[:, 1], 'k.', markersize=3, color=color, label=lab)

plt.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],'k.', markersize=10, color='white', label='Centroid')
plt.title('2D Plot of K-means Clustering of MNIST with {} Centroids'.format(num_centroids))
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.legend(shadow=True).get_frame().set_facecolor('grey')
plt.show()





