'''rbf_net.py
Radial Basis Function Neural Network
CHANDRACHUD MALALI GOWDA
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import kmeans


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

        # Define number of hidden units as an instance variable called `k` (as in k clusters)
        self.k = num_hidden_units
        
        # Define number of classes (number of output units in network) as an instance variable
        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes


    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        num_clusters = centroids.shape[0]
        avg_dists = np.zeros(num_clusters)
        for i in range(num_clusters):
            cluster_data = data[cluster_assignments == i]
            centroid = centroids[i]
            # Calculate Euclidean distance between each data point and centroid of cluster
            dists = np.linalg.norm(cluster_data - centroid, axis=1)
            # Compute the average distance within the cluster
            avg_dists[i] = np.mean(dists)

        return avg_dists

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        # Perform K-means clustering
        kmeans_obj = kmeans.KMeans(data)
        kmeans_obj.cluster_batch(k=self.k, n_iter=5)
        
        # Set the prototypes as the centroids returned by K-means
        self.prototypes = kmeans_obj.get_centroids()
        
        # Determine the average distance between each cluster center and data points assigned to it
        cluster_assignments = kmeans_obj.get_data_centroid_labels()
        self.sigmas = self.avg_cluster_dist(data, self.prototypes, cluster_assignments, kmeans_obj)
        
        # Set weights self.wts using a Gaussian distribution with mean 0 and standard deviation 1.
        # self.wts = np.random.normal(loc=0, scale=1, size=(self.k+1, self.num_classes))

    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''

        # Add a column of 1s to the input matrix to represent the intercept term
        Ahat = np.hstack((np.ones((A.shape[0], 1)), A))

        # Solve the linear system using least squares method
        c, _, _, _ = np.linalg.lstsq(Ahat, y, rcond=None)

        # Extract the slope coefficients and intercept term from the solution vector
        intercept = c[0]
        slopes = c[1:].reshape(A.shape[1])

        # Return the linear regression coefficients as a single array
        return np.hstack((slopes, intercept))




    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        dist = np.zeros((data.shape[0], self.k))
        for j in range(self.k):
            diff = data - self.prototypes[j]
            dist[:, j] = np.sum(diff ** 2, axis=1)
        act = np.exp(-dist / (2 * self.sigmas ** 2))
        return act


    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        acc = self.wts[-1] + hidden_acts@self.wts[:-1]

        return acc
    
    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data)

        hidd_acc = self.hidden_act(data)

        c = np.empty((self.k+1,self.num_classes))

        for i in range(self.num_classes):
            weight = self.linear_regression(hidd_acc, np.array(i==y).astype(int)).T
            c[:,i] = weight

        self.wts = c

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        # Pass data through network
        hidden_acts = self.hidden_act(data)
        output_acts = self.output_act(hidden_acts)

        # Get predicted class for each sample
        y_pred = np.argmax(output_acts, axis=1)
        return y_pred

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        return np.mean(y == y_pred)