import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy.io import loadmat

PER = 0.05

def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    plt.plot(circles[0,:], circles[1,:], '.k')
    plt.show()

def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()

def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()

def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    return euclidean_distances(X, Y)

def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    return np.mean(X, 0)

def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """

    centers = np.array([X[np.random.randint(X.shape[0]),:]])
    for i in range(k-1):
        tempD = metric(X,centers)
        minimums = np.amin(tempD, axis=1)
        D = np.multiply(minimums,minimums).astype(np.float)
        D /= np.sum(D)
        center = X[np.random.choice(X.shape[0], p=D),:]
        centers = np.vstack((centers, center))
    return centers.astype(np.float)

def silhouette(X, clusterings):
    """
    Given results from clustering with K-means, return the silhouette measure of
    the clustering.
    :param X: The NxD data matrix.
    :param clustering: A list of N-dimensional vectors, each representing the
                clustering of one of the iterations of K-means.
    :param centroids: A list of kxD centroid matrices, one for each iteration.
    :return: The Silhouette statistic, for k selection.
    """
    a = np.zeros(X.shape[0])
    b = np.zeros(X.shape[0])
    s = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        sameClust = np.where(clusterings == clusterings[i])[0]
        diffClust = np.where(clusterings != clusterings[i])[0]
        dist = euclidean_distances(X[i, :].reshape((1, X.shape[1])), X[sameClust, :])
        a_i = np.sum(dist) / sameClust.shape[0]
        a[i] = (a_i)
        b_i = np.inf
        visitedIdx = []
        for j in diffClust:
            if clusterings[j] in visitedIdx:
                continue
            else:
                visitedIdx.append(clusterings[j])
            idx = np.where(clusterings == clusterings[j])[0]
            tempDist = euclidean_distances(X[i, :].reshape((1, X.shape[1])), X[idx])
            tempB = np.sum(tempDist) / idx.shape[0]
            if tempB < b_i:
                b_i = tempB
        b[i] = (b_i)
        s[i] = (b_i - a_i) / max(a_i, b_i)
    return np.mean(s)

def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init, stat=silhouette):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """

    optimal_labels = np.zeros(X.shape[0])
    optimal_score = np.inf
    for i in range(iterations):
        centroids = init(X, k, metric)
        labels = np.zeros(X.shape[0])
        prev_labels = np.ones(X.shape[0])
        while not np.array_equal(labels, prev_labels):
            prev_labels = labels
            D = metric(X, centroids)
            labels = np.argmin(D, axis=1)
            for c in range(k):
                centroids[c,:] = center(X[np.where(labels==c)[0],:])
        score = np.sum(euclid(X, centroids)[np.arange(X.shape[0]),labels])
        if score < optimal_score:
            optimal_score = score
            optimal_labels = labels
    return optimal_labels, centroids, optimal_score, stat(X, optimal_labels)

def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.exp(-(euclid(X, X) ** 2) / (2 * sigma ** 2))

def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    S = euclidean_distances(X, X)
    np.fill_diagonal(S,np.inf)
    idx = np.argpartition(S, m)[:,:m]
    rows = np.indices((S.shape))[0][:,:m]
    W = np.zeros(S.shape)
    W[rows, idx] = 1
    tempW = W.astype(np.bool_)
    transW = tempW.T
    W = np.logical_or(transW,tempW)
    W = W.astype(np.float)
    return W

def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    W = similarity(X, similarity_param)
    D = np.diag(np.power(np.sum(W, axis=1), -0.5))
    L = np.eye(D.shape[0])-D.dot(W).dot(D)
    w, v = np.linalg.eigh(L)
    U = v[:,np.argsort(w)][:,:k]
    # TODO normalize the matrix
    updateU = U/U.sum(axis=1)[:,np.newaxis]
    return kmeans(updateU, k, 10)

def Q1_a():
    """
    K-means on synthetic data, that I've created.
    :return: None
    """
    cov = [[0.1,0],[0,0.11]]
    x1, y1 = np.random.multivariate_normal([0,0], cov, 500).T
    x2, y2 = np.random.multivariate_normal([5,0], cov, 500).T
    x3, y3 = np.random.multivariate_normal([0,5], cov, 500).T
    x4, y4 = np.random.multivariate_normal([5,5], cov, 500).T
    x5, y5 = np.random.multivariate_normal([2.5,2.5], cov, 500).T
    x=np.concatenate((x1,x2,x3,x4,x5))
    y=np.concatenate((y1,y2,y3,y4,y5))
    X = np.vstack((x,y)).T
    scores = []
    sils = []
    K = list(range(2,8))
    plt.figure(1)
    plt.suptitle("K-means++ on synthetic data", fontsize=20)
    for k in range(len(K)):
        result = kmeans(X, K[k])
        scores.append(result[2])
        sils.append(result[3])
        plt.subplot(230+k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(X[:, 0], X[:, 1], c=result[0], cmap="gist_rainbow")
    plt.savefig("K-means++ on synthetic data")
    plt.show()

    # elbow method
    plt.figure()
    plt.plot(K, scores)
    plt.title("K-means++ Elbow Method - Synthetic Data", fontsize=15)
    plt.savefig("K-means++ Elbow Method - Synthetic Data")
    plt.show()

    # Silhouette  method
    plt.figure()
    plt.plot(K, sils)
    plt.title("K-means++ Silhouette Method - Synthetic Data", fontsize=15)
    plt.savefig("K-means++ Silhouette Method - Synthetic Data")
    plt.show()

def Q1_b():
    """
    K-means++ on the given apml pickle data
    :return: None
    """
    with open('APML_pic.pickle', 'rb') as f:
        apml = pickle.load(f)

    scores = []
    sils = []
    K = list(range(4,12))
    plt.figure(1)
    plt.suptitle("K-means++ on APML text", fontsize=18)
    for k in range(len(K)):
        result = kmeans(apml, K[k])
        scores.append(result[2])
        sils.append(result[3])
        plt.subplot(240+k+1)
        plt.title("k="+str(K[k]) , fontsize=12)
        plt.scatter(apml[:, 0], apml[:, 1], c=result[0], cmap="gist_rainbow")
    plt.savefig("K-means++ on APML text")
    plt.show()

    # elbow method
    plt.figure()
    plt.plot(K, scores)
    plt.title("K-means++ Elbow Method - APML text", fontsize=15)
    plt.savefig("K-means++ Elbow Method - APML text")
    plt.show()

    # Silhouette  method
    plt.figure()
    plt.plot(K, sils)
    plt.title("K-means++ Silhouette Method - APML text", fontsize=15)
    plt.savefig("K-means++ Silhouette Method - APML text")
    plt.show()

def Q1_c():
    """
    K-means++ on the given circles pickle data
    :return: None
    """
    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    data = np.squeeze(np.asarray(circles.T))
    scores = []
    sils = []
    K = list(range(3,7))
    plt.figure(1)
    plt.suptitle("K-means++ on circles", fontsize=20)
    for k in range(len(K)):
        result = kmeans(data, K[k])
        scores.append(result[2])
        sils.append(result[3])
        plt.subplot(220+k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(data[:, 0], data[:, 1], c=result[0], cmap="gist_rainbow")
    plt.savefig("K-means++ on circles")
    plt.show()

    # elbow method
    plt.figure()
    plt.plot(K, scores)
    plt.title("Chossing k - Elbow Method - circles", fontsize=15)
    plt.savefig("Elbow method")
    plt.show()

    # Silhouette  method
    plt.figure()
    plt.plot(K, sils)
    plt.title("Chossing k - Silhouette Method - circles", fontsize=15)
    plt.savefig("Silhouette method")
    plt.show()

def Q1():
    Q1_a()
    Q1_b()
    Q1_c()

def choose_alpha(data):
    """
    :param data: The given data
    :return: alpha for th gaussian kernel
    """
    distances = euclid(data, data)  # get distances of the data
    hist = np.histogram(distances.ravel(), bins=100)    # get histogram of distances
    cum = np.cumsum(hist[0])    # get cumulative sum of the histogram
    c= cum/cum[-1]              # normalize before taking the 0.05 percentile
    index = np.argmax(c>0.05)   # get the index that represent the 0.05 percentile
    return hist[1][index+1]     # get the distance assigned that index

def Q2_shalom(data):
    scores = []
    sils = []
    alpha = choose_alpha(data)
    K = list(range(3,11))
    plt.figure(1)
    plt.suptitle("Spectral Clustering on SHALOM text", fontsize=20)
    for k in range(len(K)):
        result = spectral(data, K[k])
        scores.append(result[2])
        sils.append(result[3])
        plt.subplot(240+k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(data[:, 0], data[:, 1], c=result[0], cmap="gist_rainbow")
    plt.savefig("Spectral Clustering on SHALOM text")
    plt.show()

    # elbow method
    plt.figure()
    plt.plot(K, scores)
    plt.title("Spectral Clustering - Elbow Method - SHALOM text", fontsize=15)
    plt.savefig("Spectral Clustering - Elbow Method - SHALOM text")
    plt.show()

    # Silhouette  method
    plt.figure()
    plt.plot(K, sils)
    plt.title("Spectral Clustering - Silhouette Method - SHALOM text", fontsize=15)
    plt.savefig("Spectral Clustering - Silhouette Method - SHALOM text")
    plt.show()

def Q2_circles(data):
    scores = []
    sils = []
    K = list(range(3,7))
    plt.figure(1)
    plt.suptitle("Spectral Clustering on circles", fontsize=20)
    for k in range(len(K)):
        result = spectral(data, K[k])
        scores.append(result[2])
        sils.append(result[3])
        plt.subplot(220+k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(data[:, 0], data[:, 1], c=result[0], cmap="gist_rainbow")
    plt.savefig("Spectral Clustering on circles")
    plt.show()

    # elbow method
    plt.figure()
    plt.plot(K, scores)
    plt.title("Spectral Clustering - Elbow Method - circles", fontsize=15)
    plt.savefig("Spectral Clustering - Elbow Method - circles")
    plt.show()

    # Silhouette  method
    plt.figure()
    plt.plot(K, sils)
    plt.title("Spectral Clustering - Silhouette Method - circles", fontsize=15)
    plt.savefig("Spectral Clustering - Silhouette Method - circles")
    plt.show()

def Q2_apml(data):
    scores = []
    sils = []
    K = list(range(3,11))
    plt.figure(1)
    plt.suptitle("Spectral Clustering on APML text", fontsize=20)
    for k in range(len(K)):
        result = spectral(data, K[k])
        scores.append(result[2])
        sils.append(result[3])
        plt.subplot(240+k+1)
        plt.title("k="+str(K[k]) , fontsize=15)
        plt.scatter(data[:, 0], data[:, 1], c=result[0], cmap="gist_rainbow")
    plt.savefig("Spectral Clustering on APML text")
    plt.show()

    # elbow method
    plt.figure()
    plt.plot(K, scores)
    plt.title("Spectral Clustering - Elbow Method - APML text", fontsize=15)
    plt.savefig("Spectral Clustering - Elbow Method - APML text")
    plt.show()

    # Silhouette  method
    plt.figure()
    plt.plot(K, sils)
    plt.title("Spectral Clustering - Silhouette Method - APML text", fontsize=15)
    plt.savefig("Spectral Clustering - Silhouette Method - APML text")
    plt.show()

def Q2():
    """
    Spectral Clustering on the given
    :return:
    """
    shalom = loadmat("d3.mat")["d3"]
    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    data = np.squeeze(np.asarray(circles.T))
    with open('APML_pic.pickle', 'rb') as f:
        apml = pickle.load(f)
    Q2_shalom(shalom)
    Q2_circles(data)
    Q2_apml(apml)



def Q6(frac = 1):
    digits = load_digits()
    if frac!=1:
        SIZE = int(frac*digits.data.shape[0])
        indices = np.random.permutation(np.arange(digits.data.shape[0]))[:SIZE]
        tsne = TSNE(n_components=2, method='exact', n_iter=300).fit_transform(digits.data[indices])
        labels = digits.target[indices]
    else:
        tsne = TSNE(n_components=2, method='exact', n_iter=300).fit_transform(digits.data)
        labels = digits.target
    plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(10):
        plt.scatter(tsne[np.where(labels==i), 0], tsne[np.where(labels==i), 1], color=colors[i], label=str(i))#, cmap="gist_rainbow")
    title= "t-SNE about MNIST - 2D"
    plt.title(title, fontsize=15)
    plt.legend(title="color2num", loc='center left', bbox_to_anchor=(1, 0.2))
    plt.savefig(title)
    plt.show()
    # plt.gray()
    # plt.matshow(digits.images[4])
    # plt.show()

if __name__ == '__main__':

    # apml_pic_example()
    # circles_example()
    # Q1()
    Q2()
    # Q3()
    # Q4()
    # Q6()
