'''kmeans.py
Performs K-Means clustering
CHANDRACHUD MALALI GOWDA
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import seaborn as sns
from sklearn.metrics import silhouette_score

class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # silhouette_score: float.
        #   Silhouette score of the clustering
        self.silhouette_score = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.linalg.norm(pt_1 - pt_2)

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.sum((centroids - pt)**2, axis=1))

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k

        # randomly select k unique indices from the data array
        centroid_indices = np.random.choice(self.num_samps, size=k, replace=False)
        
        # use the selected indices to obtain the initial centroids
        self.centroids = self.data[centroid_indices, :]
        
        return self.centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, isSilhouette=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        
        # - Initialize K-means variables
        # - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        # difference between the previous and current centroid values is > `tol`
        # - Set instance variables based on computed values.
        # (All instance variables defined in constructor should be populated with meaningful values)
        # - Print out total number of iterations K-means ran for

        # Initialize K-means variables
        self.initialize(k)
        self.data_centroid_labels = np.zeros(self.num_samps, dtype=int) # Initialize as integer array
        self.inertia = np.inf
        self.silhouette_score = np.inf
        prev_centroids = np.zeros(self.centroids.shape)
        num_iter = 0

        # Do K-means as long as the max number of iterations is not met AND the absolute value of the
        # difference between the previous and current centroid values is > `tol`
        while np.sum(np.abs(self.centroids - prev_centroids)) > tol and num_iter < max_iter:
            prev_centroids = self.centroids.copy()
            self.data_centroid_labels = self.update_labels(self.centroids) # Use updated centroids to assign labels
            self.centroids, centroid_diff = self.update_centroids(k, self.data_centroid_labels, prev_centroids) # Update centroids
            num_iter += 1

        # Set instance variables based on computed values.
        # (All instance variables defined in constructor should be populated with meaningful values)
        self.data_centroid_labels = self.update_labels(self.centroids)
        if isSilhouette:
            self.silhouette_score = self.compute_silhouette_score()
        else:
            self.inertia = self.compute_inertia()

        # Print out total number of iterations K-means ran for
        if verbose:
            print('K-means ran for {} iterations'.format(num_iter))

        if isSilhouette:
            return self.silhouette_score, num_iter
        else:
            return self.inertia, num_iter
    

    def cluster_batch(self, k=2, n_iter=1, verbose=False, isSilhouette=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        # Initialize variables
        best_output = np.inf
        best_centroids = None
        best_labels = None

        # Run K-means `n_iter` times
        for i in range(n_iter):
            output, _ = self.cluster(k, verbose=verbose, isSilhouette=isSilhouette)
            if output < best_output:
                best_output = output
                best_centroids = self.centroids.copy()
                best_labels = self.data_centroid_labels.copy()

        # Set instance variables based on best K-means run
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_output

    # Performs clustering on the data using the leader algorithm
    def cluster_leader(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data with the leader algorithm

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for
        '''

        # Using the leader algorithm to perform K-means clustering
        # - Initialize K-means variables
        # - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        # difference between the previous and current centroid values is > `tol`
        # - Set instance variables based on computed values.
        # (All instance variables defined in constructor should be populated with meaningful values)
        # - Print out total number of iterations K-means ran for

        # Initialize K-means variables
        self.initialize(k)
        self.data_centroid_labels = np.zeros(self.num_samps, dtype=int) # Initialize as integer array
        self.inertia = np.inf
        self.silhouette_score = np.inf
        prev_centroids = np.zeros(self.centroids.shape)
        num_iter = 0

        # Do K-means as long as the max number of iterations is not met AND the absolute value of the
        # difference between the previous and current centroid values is > `tol`
        while np.sum(np.abs(self.centroids - prev_centroids)) > tol and num_iter < max_iter:
            prev_centroids = self.centroids.copy()
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, centroid_diff = self.update_centroids_leader(k, self.data, self.centroids)
            num_iter += 1

        # Set instance variables based on computed values.
        # (All instance variables defined in constructor should be populated with meaningful values)
        self.data_centroid_labels = self.update_labels(self.centroids)
        self.inertia = self.compute_inertia()

        # Print out total number of iterations K-means ran for
        if verbose:
            print('K-means ran for {} iterations'.format(num_iter))


        return self.inertia, num_iter

    def update_centroids_leader(self, k, data, centroids):
        '''Updates the centroids based on the leader algorithm

        Parameters:
        -----------
        k: int. Number of clusters
        data: ndarray. shape=(self.num_samps, self.num_features). Holds the data samples.
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Updated centroids for the k clusters.
        float. The (absolute value of the) difference between all the centroid values from the
        previous and current time step.
        '''
        # Initialize variables
        new_centroids = np.zeros(centroids.shape)
        centroid_diff = 0

        # Update each centroid
        for i in range(k):
            # Get the data samples that belong to cluster i
            data_in_cluster = data[self.data_centroid_labels == i]

            # Update the centroid for cluster i
            new_centroids[i] = np.mean(data_in_cluster, axis=0)

        # Compute the difference between the previous and current centroid values
        centroid_diff = np.sum(np.abs(new_centroids - centroids))

        return new_centroids, centroid_diff


    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        # Calculate the Euclidean distance between each data point and all centroids
        distances = np.sqrt(((self.data - centroids[:, np.newaxis])**2).sum(axis=2))

        # Assign each data point to the nearest centroid
        labels = np.argmin(distances, axis=0)

        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = np.zeros((k, self.num_features))
        centroid_counts = np.zeros(k)

        # Compute new centroids
        for i in range(self.num_samps):
            label = data_centroid_labels[i]
            new_centroids[label] += self.data[i]
            centroid_counts[label] += 1

        # Handle empty clusters
        empty_clusters = np.where(centroid_counts == 0)[0]
        for i in empty_clusters:
            idx = np.random.choice(self.num_samps)
            while idx in data_centroid_labels:
                idx = np.random.choice(self.num_samps)
            new_centroids[i] = self.data[idx]
            centroid_counts[i] = 1

        # Normalize by number of points in each cluster
        for i in range(k):
            if centroid_counts[i] > 0:
                new_centroids[i] /= centroid_counts[i]

        centroid_diff = new_centroids - prev_centroids


        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        distances = np.zeros(self.num_samps)
        for i in range(self.num_samps):
            centroid = self.centroids[self.data_centroid_labels[i]]
            distances[i] = np.sum((self.data[i] - centroid) ** 2)
        inertia = np.mean(distances)
        return inertia
    
    # Compute silhouette score for number-of-clusters detection metrics
    def compute_silhouette_score(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        # COmpute the silhouette score
        score = silhouette_score(self.data, self.data_centroid_labels, metric='euclidean')
        return score
        
    
    def plot_clusters(self, isSilhouette=False):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
       
        #    - Plot samples belonging to a cluster with the same color.
        #     - Plot the centroids in black with a different plot marker.
        #     - The default scatter plot color palette produces colors that may be difficult to discern
        #     (especially for those who are colorblind). Make sure you change your colors to be clearly
        #     differentiable.
        #         You should use a palette Colorbrewer2 palette. Pick one with a generous
        #         number of colors so that you don't run out if k is large (e.g. 10).
        #     '''

        plt.figure(figsize=(10, 10))
        color_list = cartocolors.qualitative.Vivid_10.mpl_colors
        for i in range(self.k):
            w = self.get_data()[self.data_centroid_labels == i]
            plt.scatter(w[:, 0], w[:, 1], color = color_list[i], label='cluster ' + str(i))

        centroids_without_clusters = []
        for j in range(self.k):
            if j in self.data_centroid_labels:
                centroids_without_clusters.append(j)
        plt.scatter(self.centroids[centroids_without_clusters,0], self.centroids[centroids_without_clusters,1], color = color_list[-1], label='centroids', marker = '*')
        
        if isSilhouette:
            plt.title('Data by ' + str(self.k) + ' clusters - silhouette score: ' + str(round(self.silhouette_score,3)))
        else:
            plt.title('Data by ' + str(self.k) + ' clusters - inertia: ' + str(round(self.inertia,3)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(bbox_to_anchor=(1, 0),loc='lower left', fontsize='small')
       

    def elbow_plot(self, max_k, n_iter=1, isSilhouette=False):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        outputs = []
        for i in range(1, max_k+1):
            self.initialize(i)
            self.cluster_batch(i, n_iter = n_iter, isSilhouette = isSilhouette)
            output = None
            if isSilhouette:
                output = self.compute_silhouette_score()
            else:
                output = self.compute_inertia()
            outputs.append(output)
        plt.plot(range(1, max_k+1), outputs, 'o-')
        plt.xticks(range(1, max_k+1))
        plt.xlabel('Number of clusters')
        if isSilhouette:
            plt.ylabel('Silhouette score')
            plt.title('Elbow plot showing silhouette score under different k values')
        else:
            plt.ylabel('Inertia')
            plt.title('Elbow plot showing inertia under different k values')
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''

        data = []
        for i in range(self.num_samps):
            centroid = np.rint(self.centroids[self.data_centroid_labels[i]]).astype(int)
            data.append(centroid)

        self.set_data(np.array(data))

