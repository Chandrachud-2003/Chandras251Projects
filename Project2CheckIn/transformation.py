'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 252 Data Analysis Visualization, Spring 2022
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''

        # Pass `data` to the superclass constructor.
        super().__init__(data)

        # Create an instance variable for `orig_dataset`.
        self.orig_dataset = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''

        # Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        # variables). Determine and fill in 'valid' values for all the `Data` constructor
        # keyword arguments (except you dont need `filepath` because it is not relevant here).

        # Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        # variables). Determine and fill in 'valid' values for all the `Data` constructor
        # keyword arguments (except you dont need `filepath` because it is not relevant here).
        # Modifying the header2col to only contain the headers that are in the headers list and changing the values to the corresponding values in the headers list.
        header2col = {header.strip(): i for i, header in enumerate(headers)}
        self.data = data.Data(filepath = "", headers=headers, data=self.orig_dataset.select_data(headers = headers), header2col=header2col)
        
    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''

        # TODO: Return a version of the projected data array with an added homogeneous coordinate.
        # Useful for homogeneous transformations.
        return np.hstack((self.data.get_all_data(), np.ones((self.data.get_all_data().shape[0], 1))))

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        ''' 

        # Setting the other magnitudes to 0 if the length of the magnitudes is less than the number of dimensions
        if len(magnitudes) < self.data.get_num_dims():
            magnitudes = magnitudes + [0] * (self.data.get_num_dims() - len(magnitudes))
        translation_matrix = np.eye(self.data.get_num_dims()+1)

        translation_matrix[:-1, -1] = magnitudes
        return translation_matrix

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''

        if len(magnitudes) < self.data.get_num_dims():
            magnitudes = magnitudes + [0] * (self.data.get_num_dims() - len(magnitudes))
        transformation_matrix = np.eye(self.data.get_num_dims()+1)
        transformation_matrix[:-1, :-1] = np.diag(magnitudes)
        return transformation_matrix

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        # create the translation matrix
        translation_matrix = self.translation_matrix(magnitudes)

        # apply the translation matrix to the data
        data_translated = self.data.get_all_data() @ translation_matrix[:-1,:-1] + translation_matrix[:-1,-1]

        # create a new Data object with the translated data
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        new_data = data.Data(headers = headers, header2col = header2col, data = data_translated)

        # update self.data with the new Data object
        self.data = new_data

        return data_translated
    
    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        # create the scaling matrix
        scaling_matrix = self.scale_matrix(magnitudes)

        # apply the scaling matrix to the data
        data_scaled = self.data.get_all_data() @ scaling_matrix[:-1,:-1] + scaling_matrix[:-1,-1]

        # create a new Data object with the scaled data
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        new_data = data.Data(headers = headers, header2col = header2col, data = data_scaled)

        # update self.data with the new Data object
        self.data = new_data

        return data_scaled

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        # Use matrix multiplication to apply the compound transformation matix `C` to the projected
        # dataset.
        # Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        # dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        # transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        # coordinate!

        # apply the transformation matrix to the data
        data_transformed = self.data.get_all_data() @ C[:-1,:-1] + C[:-1,-1]

        # create a new Data object with the transformed data
        headers = self.data.get_headers()
        header2col = self.data.get_mappings()
        new_data = data.Data(headers = headers, header2col = header2col, data = data_transformed)

        # update self.data with the new Data object
        self.data = new_data

        return data_transformed

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        # Find the global minimum and range across all variables
        data_array = self.data.get_all_data()
        global_min = np.min(data_array)
        global_range = np.max(data_array) - global_min
        
        # Translate the global minimum to zero
        data_array -= global_min
        
        # Scale the global range to one
        data_array /= global_range
        
        # Return the normalized data
        return data_array


    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        # Find the local minimum and range for each variable
        data_array = self.data.get_all_data()
        local_min = np.min(data_array, axis=0)

        local_range = np.max(data_array, axis=0) - local_min

        # Translate the local minimum to zero
        data_array -= local_min

        # Scale the local range to one
        data_array /= local_range

        # Return the normalized data
        return data_array

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        # Get the column index of the variable `header`
        col = self.data.header2col[header]

        # Create the rotation matrix
        R = np.eye(4)
        R[col, col] = np.cos(np.deg2rad(degrees))

        # Return the rotation matrix
        return R

        pass

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        # Use matrix multiplication to rotate the projected dataset, as advertised above.
        # Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        # dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        # transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        # homogenous coordinate!

        self.data = data.Data(headers=self.data.get_headers(), data=np.dot(self.get_data_homogeneous(), self.rotation_matrix_3d(header, degrees))[:, :-1])
        return self.data.get_all_data()

    def scatter3d(self, xlim, ylim, zlim, better_view=False):
        '''Creates a 3D scatter plot to visualize data the x, y, and z axes are drawn, but not ticks

        Axis labels are placed next to the POSITIVE direction of each axis.

        Parameters:
        -----------
        xlim: List or tuple indicating the x axis limits. Format: (low, high)
        ylim: List or tuple indicating the y axis limits. Format: (low, high)
        zlim: List or tuple indicating the z axis limits. Format: (low, high)
        better_view: boolean. Change the view so that the Z axis is coming "out"
        '''
        if len(self.data.get_headers()) != 3:
            print("need 3 headers to make a 3d scatter plot")
            return

        headers = self.data.get_headers()
        xyz = self.data.get_all_data()

        if better_view:
            # by default, matplot lib puts the 3rd axis heading up
            # and the second axis heading back.
            # rotate it so that the second axis is up and the third is forward
            R = np.eye(3)
            R[1, 1] = np.cos(np.pi/2)
            R[1, 2] = -np.sin(np.pi/2)
            R[2, 1] = np.sin(np.pi/2)
            R[2, 2] = np.cos(np.pi/2)
            xyz = (R @ xyz.T).T

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Scatter plot of data in 3D
        ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.plot(xlim, [0, 0], [0, 0], 'k')
        ax.plot([0, 0], ylim, [0, 0], 'k')
        ax.plot([0, 0], [0, 0], zlim, 'k')
        ax.text(xlim[1], 0, 0, headers[0])

        if better_view:
            ax.text( 0, zlim[0], 0, headers[2])
            ax.text( 0, 0, ylim[1], headers[1])
        else:
            ax.text(0, ylim[1], 0, headers[1])
            ax.text(0, 0, zlim[1], headers[2])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        plt.show()

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        # Get the data for the variable `ind_var`
        ind_data = self.data.select_data([ind_var])

        # Get the data for the variable `dep_var`
        dep_data = self.data.select_data([dep_var])

        # Get the data for the variable `c_var`
        c_data = self.data.select_data([c_var])

        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Use a ColorBrewer color palette to implement the color scale (e.g. from the `palettable` library).
        # - To do so, go to https://jiffyclub.github.io/palettable/colorbrewer/, and examine maps in the 3 categories (diverging, qualitative, and sequential) to find the appropriate map. Access via the naming scheme below. The map has an attribute named mpl_colormap that can be passed in to `scatter` to control the colors (as the value for the cmap parameter). We use the third feature (which we are calling Z here) to determine which values of the color map are used for which data points. We can also control the outline of the points with the edgecolor argument (here we make it black).
        #         color_map = palettable.colorbrewer.sequential.Purples_9
        #         scatter(X, Y, c=Z, s=75, cmap=color_map.mpl_colormap, edgecolor='black')
        
        # Import the palettable library
        import palettable

        # Create a color map using the palettable library
        color_map = palettable.colorbrewer.sequential.Purples_9

        # Create a scatter plot of the data
        ax.scatter(ind_data, dep_data, c=c_data, cmap=color_map.mpl_colormap, edgecolor='black')

        # Set the title of the figure
        ax.set_title(title)

        # Set the labels of the axes
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)

        # Set the color bar label
        cbar = ax.figure.colorbar(ax.collections[0])

        # Adding space between the color bar aand color bar label
        cbar.ax.yaxis.set_ticks_position('right')

        # Show the figure
        plt.show()

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
