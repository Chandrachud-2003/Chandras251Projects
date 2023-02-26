'''analysis.py
Run statistical analyses and plot Numpy ndarray data
YOUR NAME HERE
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''
        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''

        # Assign the data object to the instance variable
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        mins = np.zeros(len(headers))
        mins = np.min(self.data.select_data(headers, rows), axis=0)

        return mins

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        
        maxs = np.zeros(len(headers))
        maxs = np.max(self.data.select_data(headers, rows), axis=0)

        return maxs

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''

        mins = self.min(headers, rows)
        maxes = self.max(headers, rows)

        # Return the mins and maxes
        return mins, maxes


    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''

        # Computing the sum of the data samples for each variable
        sum = np.sum(self.data.select_data(headers, rows), axis=0)
        # Computing the mean of the data samples for each variable
        means = sum / len(self.data.select_data(headers, rows))

        return means

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''

        # Computing the means of the data samples for each variable
        means = self.mean(headers, rows)
        # Getting the difference between the data samples and the mean
        diff = self.data.select_data(headers, rows) - means
        # Computing the variance of the data samples for each variable
        vars = np.sum(diff**2, axis=0) / (len(self.data.select_data(headers, rows)) - 1)

        # Return the variance
        return vars

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''

        # Loop through each header variable name in `headers`
        # Compute the standard deviation value for that variable in the data object
        # Store the standard deviation value in the `stds` array

        # Computing the difference between the data samples and the mean
        diff = self.data.select_data(headers, rows) - self.mean(headers, rows)
        # Computing the standard deviation of the data samples for each variable
        stds = np.sqrt(np.sum(diff**2, axis=0) / (len(self.data.select_data(headers, rows)) - 1))

        # Return the standard deviation
        return stds

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''

        # Get the x and y values from the ind_var and dep_var columns in the data object
        # Both 'ind_var' and 'dep_var' should be strings in `self.headers`
        # Use the select_rows method to get the x and y values
        x = self.data.select_data([ind_var], [])
        y = self.data.select_data([dep_var], [])

        # Flatten the x and y arrays
        x = x.flatten()
        y = y.flatten()

        cmap = plt.cm.get_cmap('plasma')

        # Create the scatter plot
        plt.scatter(x, y, c=y, cmap=cmap)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)

        plt.colorbar(label=dep_var)

        return x, y

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''

        cmap = plt.cm.get_cmap('plasma')

        # Make the len(data_vars) x len(data_vars) grid of scatterplots
        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize=fig_sz, sharex='col', sharey='row')
        # Creating scatter plots for each combination of variables in `data_vars`
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                # Creating the scatter plot
                ax = axes[i, j]
                x = self.data.select_data([data_vars[j]], [])
                y = self.data.select_data([data_vars[i]], [])
                ax.scatter(x, y, c=y, cmap=cmap)
                # Setting the x and y labels
                if i == len(data_vars) - 1:
                    ax.set_xlabel(data_vars[j])
                if j == 0:
                    ax.set_ylabel(data_vars[i])
                # Removing the x and y ticks
                ax.set_xticks([])
                ax.set_yticks([])

        # Setting the title
        fig.suptitle(title)

        # Setting a common colorbar
        fig.colorbar(ax.collections[0], ax=axes, label=data_vars[-1])

        # Return the figure and axes
        return fig, axes

        

