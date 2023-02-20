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

    # Extension methods:
    

    # Boxplot
    def boxplot(self, headers, rows=[]):
        '''Creates a side-by-side boxplot for each variable in `headers` in the data object.
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
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(headers),)
            2nd item returned by plt.subplots
        '''

        # Creating the figure and axes
        fig, axes = plt.subplots(1, len(headers), figsize=(len(headers) * 4, 4))
        
        # Creating a color map based on y values
        cmap = plt.get_cmap('cool')
        colors = cmap(np.linspace(0, 1, len(headers)))

        # Looping through each header variable name in `headers`
        for i in range(len(headers)):
            # Creating the boxplot with different colors
            ax = axes[i]
            data = self.data.select_data([headers[i]], rows)
            bp = ax.boxplot(data, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[i])
                
            # Setting the x and y labels
            ax.set_xlabel(headers[i])
            ax.set_ylabel('Value')
        
        # Creating a color bar to show what each color corresponds to
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(headers)))
        sm.set_array([])
        fig.colorbar(sm, ax=axes)
        
        # SETTING THE TITLE
        fig.suptitle('Boxplots for the Headers: ' + str(headers))

        return fig, axes  

    # Histograms
    def histogram(self, headers, rows=[]):
        '''Creates a histogram for each variable in `headers` in the data object.
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
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(headers),)
            2nd item returned by plt.subplots
        '''

        # Creating the figure and axes
        fig, axes = plt.subplots(1, len(headers), figsize=(len(headers) * 4, 4))
        # Looping through each header variable name in `headers`
        for i in range(len(headers)):
            # Creating the histogram
            ax = axes[i]
            counts, bins, patches = ax.hist(self.data.select_data([headers[i]], rows))
            # Setting the x and y labels
            ax.set_xlabel(headers[i])
            ax.set_ylabel('Frequency')
            # Set different colors based on the y value of the plot
            color_values = plt.cm.plasma(counts / float(counts.max()))
            for patch, color in zip(patches, color_values):
                patch.set_facecolor(color)

        # Create a color bar to show the mapping between color and value
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=counts.min(), vmax=counts.max()))
        fig.colorbar(sm, ax=axes)

        # SETTING THE TITLE
        fig.suptitle('Histograms for the Headers: ' + str(headers))

        return fig, axes

    # Heatmap
    def heatmap(self, headers, rows=[]):

        # Importing sns and pandas
        import seaborn as sns
        import pandas as pd

        '''Creates a heatmap for each variable in `headers` in the data object.
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
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(headers),)
            2nd item returned by plt.subplots
        '''

        # Creating the figure and axes
        fig, ax = plt.subplots(figsize=(len(headers) * 4, len(headers) * 4))
        
        # Converting the selected data to a Pandas DataFrame and creating the heatmap
        ax = sns.heatmap(pd.DataFrame(self.data.select_data(headers, rows)).corr(), annot=True, cmap='plasma')
        
        # Setting the title
        ax.set_title('Heatmap for the Headers: ' + str(headers))

        return fig, ax

    # 3D scatterplot
    def scatter3d(self, headers, rows=[]):
        '''Creates a 3D scatterplot for each variable in `headers` in the data object.
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
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(headers),)
            2nd item returned by plt.subplots
        '''

        # Importing Axes3D
        from mpl_toolkits.mplot3d import Axes3D

        # Creating the figure and axes
        fig = plt.figure(figsize=(len(headers) * 4, len(headers) * 4))
        ax = fig.add_subplot(111, projection='3d')
        # Creating the 3D scatterplot with colors based on y-values and plasma colormap
        y_data = self.data.select_data([headers[1]], rows)
        ax.scatter(self.data.select_data([headers[0]], rows), y_data, self.data.select_data([headers[2]], rows), c=y_data, cmap='plasma')
        # Setting the x and y labels
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[1])
        ax.set_zlabel(headers[2])
        # Setting the title
        ax.set_title('3D Scatterplot for the Headers: ' + str(headers))
        
        # Adding a colorbar for the plasma colormap
        cbar = plt.colorbar(ax.collections[0])
        cbar.ax.set_ylabel(headers[1])

        return fig, ax

    # Density plot
    def density(self, headers, rows=[]):
        '''Creates a density plot for each variable in `headers` in the data object.
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
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(headers),)
            2nd item returned by plt.subplots
        '''

        import pandas as pd

        # Creating the figure and axes
        fig, axes = plt.subplots(1, len(headers), figsize=(len(headers) * 4, 4))
        # Looping through each header variable name in `headers`
        for i in range(len(headers)):
            # Creating the density plot
            ax = axes[i]
            data = self.data.select_data([headers[i]], rows)
            pd.DataFrame(data).plot.kde(ax=ax)
            # Setting the x and y labels
            ax.set_xlabel(headers[i])
            ax.set_ylabel('Density')

        # SETTING THE TITLE
        fig.suptitle('Density Plots for the Headers: ' + str(headers))

        return fig, axes