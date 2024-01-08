import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class KMeansClustring:
    
    def __init__(self, k=2):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            size=(self.k, X.shape[1]))
        
        for _ in range(max_iterations):
            y = []

            for data_point in X:
                distances = KMeansClustring.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            
            y = np.array(y)
            
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                     cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y    

def K_means_Clustering():

    df = pd.read_csv('./dataset/creditcard.csv')
    Y = df['Class'].values
    X = df.drop(['Class', 'Time', 'Amount'], axis=1).values

    kmeans_model = KMeansClustring(k=2)

    Y_Pred = kmeans_model.fit(X)

    print("length:", len(Y_Pred), "Y pred:", Y_Pred)
    print("length:", len(Y), "Y:", Y)

 
    plt.scatter(X[:, 0], X[:, 1], c=Y_Pred, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans_model.centroids[:, 0], kmeans_model.centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
    plt.title('K-Means Clustering Results')
    plt.xlabel('combination of all attr')
    plt.ylabel('class')
    plt.legend()
    plt.show()

    count = 0
    for i, pred_label in enumerate(Y_Pred):
        if pred_label == Y[i]:
            count = count + 1

    accuracy = count / len(Y_Pred)
    print("Accuracy:", accuracy)



def Agglomerative_Clustering():
    pass

def DBSCAN_Clustring():
    pass

def main():
    K_means_Clustering()
    Agglomerative_Clustering()
    DBSCAN_Clustring()

if __name__ == "__main__":
    main()