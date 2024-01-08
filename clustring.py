# Import necessary libraries
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
    # Load the dataset
    # Replace 'your_dataset.csv' with the actual file path or URL of your dataset
    df = pd.read_csv("./dataset/creditcard.csv")

    # Drop unnecessary columns for clustering
    X = df.drop(['Class', 'Time', 'Amount'], axis=1)

    # Standardize the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Reduce dimensionality using PCA (optional, for visualization purposes)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    # Choose the number of clusters (k)
    k = 2  # You may adjust this based on the nature of your dataset

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_std)

    # Visualize the results (using PCA for 2D visualization)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.5)
    plt.title('K-Means Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Assuming df is your DataFrame with 'Cluster' column from k-means
    silhouette_avg = silhouette_score(X_std, df['Cluster'])
    print(f"Silhouette Score: {silhouette_avg}")




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