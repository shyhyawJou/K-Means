from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time


"""
    ''' k-means and k-means++ '''

    my code is faster than scikit-learn when the max_iter is set to the same. 
    
    However, if you didn't set the value of max_iter, the iteration didn't stop 
    until the element within every cluster no longer change. When number of data
    is very huge such as billion, it may take some time to iterate until the
    element within every cluster no longer change.

    So when data is very big, set the max_iter is a good choice.

    '''''''
    Argument:
        1. data:     should be 2d array with shape (n_samples, n_features in every sample)
        2. k:        num of cluster
        3. max_iter: max iteration
        4. seed:     numpy random seed, if you want to reproduce the result, please set a value

    return: 
        1. centroid: the centroid of every cluster, 2d array (n_cluster, n_features)
        2. label:    the cluster label to which every data sample belong
        3. iter_num: how many iteration to get the result 
"""

def k_means(data, k, max_iter=None, seed=None):
    # whether to set seed
    if seed is not None:
        np.random.seed(seed)
    # initialization
    data = np.asanyarray(data, dtype=np.float32)
    sample_num = data.shape[0]
    assert sample_num >= k
    centroid = data[np.random.permutation(sample_num)[:k]]
    last_label = np.asanyarray(None)
    iter_num = 0
    # iteration
    while True:
        # caculate euclidean distance
        euc_dist = np.sum((data[:, None, :] - centroid[None, ...]) ** 2, axis=2)
        label = np.argmin(euc_dist, axis=1)
        iter_num += 1
        # update centroid or end program
        if np.all(last_label == label):
            break
        else:
            centroid = []
            for i in range(k):
                centroid.append(np.mean(data[i == label], axis=0))
            centroid = np.asanyarray(centroid)            
            last_label = label
        # max iteration
        if iter_num == max_iter:
            break

    return centroid, label, iter_num


def k_means_plus_plus(data, k, max_iter=None, seed=None):
    # whether to set seed
    if seed is not None:
        np.random.seed(seed)
    # initialization
    data = np.asanyarray(data, dtype=np.float32)
    sample_num = data.shape[0]
    assert sample_num >= k
    last_label = np.asanyarray(None)
    iter_num = 0
    epsilon = 1e-7
    # selecting k centroid
    centroid = []
    centroid.append(data[np.random.randint(sample_num)])
    centroid = np.asanyarray(centroid)
    for _ in range(k-1):
        euc_dist = np.sum((data[:, None, :] - centroid[None, ...]) ** 2, axis=2)
        centroid = centroid.tolist()
        # get max index of harmonic mean
        centroid.append(data[np.argmin(np.mean(1 / (euc_dist + epsilon), axis=1))])
        centroid = np.asanyarray(centroid)
    # iteration
    while True:
        # caculate euclidean distance
        euc_dist = np.sum((data[:, None, :] - centroid[None, ...]) ** 2, axis=2)
        label = np.argmin(euc_dist, axis=1)
        iter_num += 1
        # update centroid or end program
        if np.all(last_label == label):
            break
        else:
            centroid = []
            for i in range(k):
                centroid.append(np.mean(data[i == label], axis=0))
            centroid = np.asanyarray(centroid)            
            last_label = label
        # max iteration
        if iter_num == max_iter:
            break

    return centroid, label, iter_num


if __name__ == "__main__":
    k = 3

    data = np.random.randint(1, 2501, size=(2500,2))
    data2 = np.random.uniform(1, 2501, size=(2500,2))
    data = np.concatenate((data, data2), 0)
    
    centroid, label, iter_num = k_means(data, k, seed=None)
    centroid2, label2, iter_num2 = k_means_plus_plus(data, k, seed=None)

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(data[label == 0][:,0], data[label == 0][:,1], c='m')
    ax[0].scatter(data[label == 1][:,0], data[label == 1][:,1], c='g')
    ax[0].scatter(data[label == 2][:,0], data[label == 2][:,1], c='r')
    #ax[0].scatter(data[label == 3][:,0], data[label == 3][:,1], c='y')
    #ax[0].scatter(data[label == 4][:,0], data[label == 4][:,1], c='c')
    ax[0].scatter(centroid[:,0], centroid[:,1], c='b', s=700, alpha=0.5)
    ax[0].set_xlabel('x', fontsize=20)
    ax[0].set_ylabel('y', fontsize=20)
    ax[0].set_title('K-Means', fontsize=22)
    
    ax[1].scatter(data[label2 == 0][:,0], data[label2 == 0][:,1], c='m')
    ax[1].scatter(data[label2 == 1][:,0], data[label2 == 1][:,1], c='g')
    ax[1].scatter(data[label2 == 2][:,0], data[label2 == 2][:,1], c='r')
    #ax[1].scatter(data[label2 == 3][:,0], data[label2 == 3][:,1], c='y')
    #ax[1].scatter(data[label2 == 4][:,0], data[label2 == 4][:,1], c='c')
    ax[1].scatter(centroid2[:,0], centroid2[:,1], c='b', s=700, alpha=0.5)
    ax[1].set_xlabel('x', fontsize=20)
    ax[1].set_ylabel('y', fontsize=20)
    ax[1].set_title('K-Means++', fontsize=22)
    plt.show()
