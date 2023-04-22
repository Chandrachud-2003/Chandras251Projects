'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
YOUR NAME HERE
CS 251/2: Data Analysis Visualization
Spring 2023
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`.
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        self.num_classes = num_classes
        self.class_priors = None
        self.log_class_likelihoods = None
        self.class_likelihoods = None
        self.log_class_priors = None

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.class_priors

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.class_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape

        class_counts = np.unique(y, return_counts=True)[1]
        priors = class_counts / num_samps
        self.class_priors = priors
        self.log_class_priors = np.log(self.class_priors)

        likelihoods = np.zeros((self.num_classes, num_features))
        for c in range(self.num_classes):
            indices = y == c
            class_data = data[indices]
            Nc = class_counts[c]
            Nc2 = class_data.sum()
            Ncw = class_data.sum(axis=0)
            likelihoods[c] = (Ncw+1) / (Nc2+num_features)

        self.class_likelihoods = likelihoods
        self.log_class_likelihoods = np.log(self.class_likelihoods)



    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `test_data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        test_data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Compute the log of the posterior by evaluating the log of the right-hand side of Bayes Rule
        without the denominator (see notebook for equation).
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability.

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''

        num_test_samps, num_features = data.shape

        predicted_classes = np.zeros(num_test_samps, dtype=int)
        for i in range(num_test_samps):
            log_class_priors = np.log(self.class_priors)
            log_class_likelihoods = np.log(self.class_likelihoods)
            log_posteriors = log_class_priors + np.dot(log_class_likelihoods, data[i])
            predicted_classes[i] = np.argmax(log_posteriors)

        return predicted_classes




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


    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        num_classes = self.num_classes
        cm = np.zeros((num_classes, num_classes))

        for i, j in zip(y, y_pred):
            cm[i, j] += 1

        return cm


