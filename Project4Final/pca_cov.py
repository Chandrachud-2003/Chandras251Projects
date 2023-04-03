'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        # Center the data
        data_centered = self.center(data)
        
        # Compute the covariance matrix
        cov = np.dot(data_centered.T, data_centered) / (data_centered.shape[0] - 1)
        
        return cov

    def center(self, data):
        '''Centers the data by subtracting the mean of each variable'''
        return data - np.mean(data, axis=0)

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        tot_var = np.sum(e_vals)
        prop_var = [(ev / tot_var) for ev in e_vals]
        return prop_var

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var = [prop_var[0]]
        for i in range(1, len(prop_var)):
            cum_var.append(cum_var[-1] + prop_var[i])
        return cum_var

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''

        # Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        data_selected = self.data[vars].to_numpy()

        # If `normalize` is True, normalize the selected data
        if normalize:
            # Compute normalization parameters for each variable
            norm_params = []
            for i in range(data_selected.shape[1]):
                min_val = np.min(data_selected[:, i])
                max_val = np.max(data_selected[:, i])
                norm_params_i = (min_val, max_val)
                norm_params.append(norm_params_i)

                # Normalize the variable (column)
                data_selected[:, i] = (data_selected[:, i] - min_val) / (max_val - min_val)

            self.norm_params = norm_params

        # Compute the covariance matrix of the selected data
        data_centered = self.center(data_selected)
        cov_mat = self.covariance_matrix(data_centered)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        e_vals, e_vecs = np.linalg.eig(cov_mat)

        # Sort the eigenvalues and eigenvectors by the eigenvalues in descending order
        sorted_indices = np.argsort(e_vals)[::-1]
        e_vals = e_vals[sorted_indices]
        e_vecs = e_vecs[:, sorted_indices]

        # Compute the proportion variance accounted for by each PC
        prop_var = self.compute_prop_var(e_vals)

        # Compute the cumulative variance accounted for by the PCs
        cum_var = self.compute_cum_var(prop_var)

        # Set the instance variables
        self.vars = vars
        self.normalized = normalize
        self.data_selected = data_selected
        self.cov_mat = cov_mat
        self.e_vals = e_vals
        self.e_vecs = e_vecs
        self.prop_var = prop_var
        self.cum_var = cum_var
        
        # Create the data matrix A
        self.A = data_selected
        
        # Store the number of variables and samples
        self.num_vars = self.A.shape[1]
        self.num_samps = self.A.shape[0]
        
        # Create the data matrix A_proj
        self.A_proj = None


    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if num_pcs_to_keep is None:
            num_pcs = len(self.prop_var)
            num_pcs_to_keep = num_pcs
        cum_var = self.compute_cum_var(self.compute_prop_var(self.e_vals))
        cum_var = cum_var[:num_pcs_to_keep]
        plt.plot(range(1, num_pcs_to_keep+1), cum_var, marker='o', markersize=8)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Proportion Variance Accounted For')

        # Printing the variance accounted for by each PC
        print('Variance accounted for by each PC:')
        for i in range(num_pcs_to_keep):
            print('PC {}: {:.2f}%'.format(i+1, cum_var[i]*100))

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        
        '''

        # Extract relevant components from covariance matrix
        components = self.e_vecs[:, pcs_to_keep]

        centered_data = self.center(self.data_selected)

        # Project the data onto the selected principal components
        pca_proj = centered_data @ components

        # Set A_proj instance variable
        self.A_proj = pca_proj

        return pca_proj


    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        # Project the data onto `top_k` PCs
        pca_proj = self.pca_project(list(range(top_k)))

        # Project back to the data space
        data_proj = pca_proj @ self.e_vecs[:, :top_k].T

        # Compute normalization parameters if necessary
        if self.normalized:
            col_mins = self.data_selected.min(axis=0)
            col_maxs = self.data_selected.max(axis=0)
            norm_ranges = col_maxs - col_mins
            norm_mins = col_mins
            data_proj = data_proj * norm_ranges + norm_mins

        return data_proj
