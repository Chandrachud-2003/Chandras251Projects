'''linear_regression.py
Subclass of Analysis that performs linear regression on data
CHANDRACHUD MALALI GOWDA
CS251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''

        # Set instance variables for independent and dependent variables
        self.ind_vars = ind_vars
        self.dep_var = dep_var

        # Extract the independent and dependent variable data from the data object
        x = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])

        # Add a column of 1s for the intercept
        self.A = np.hstack((np.ones((x.shape[0], 1)), x))

        # Fit the regression and get the coefficients
        c, _, _, _ = scipy.linalg.lstsq(self.A, self.y)

        # Set the instance variables for the slope, intercept, and residuals
        self.slope = c[1:]
        self.intercept = c[0]
        # Converting the intercept to a float
        self.intercept = float(self.intercept)

        # Compute R^2 on the fit
        y_pred = self.predict()
        # Printing the shape of y_pred
        self.R2 = self.r_squared(y_pred)

        # Compute the residuals
        self.residuals = self.compute_residuals(y_pred)

        # Setting the mse
        self.mse = self.compute_mse()
            
    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.

        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        # Use fitted linear regression model to predict the values of data matrix self.A.

        # Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        # A is the data matrix.

        # If X is None, use self.A for the "x values" when making predictions.
        # If X is not None, use X as independent var data as "x values" used in making predictions.
        if X is None:
            X = self.A
        else:
            X = np.hstack((np.ones((X.shape[0], 1)), X))


        # 'Use fitted linear regression model to predict the values of data matrix self.A.

        # Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        # A is the data matrix.

        # Reshape intercept to have the same dimensions as slope
        temp_intercept = np.reshape(self.intercept, (1, 1))

        # Combine slope and intercept to get the c vector
        c = np.vstack((temp_intercept, self.slope))
        y_pred = np.dot(X, c)


        return y_pred


    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        
        numerator = np.sum((self.y - y_pred) ** 2)
        denominator = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - (numerator / denominator)

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        
        return self.y - y_pred

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        
        # This is the equation for the mean squared error
        # **Equation for MSE:** $$E = \frac{1}{N}\sum_{i=1}^N \left (y_i - \hat{y}_i \right )^2$$
        # where $N$ is the number of data samples, $y_i$ is the actual y value at the $i^{th}$ data sample,
        # and $\hat{y}_i$ is the predicted y value at the $i^{th}$ data sample.
        mse = np.sum(self.residuals ** 2) / self.residuals.shape[0]
        return mse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''

        # Importing Analysis class
        from analysis import Analysis
        
        # Creating an analysis object
        analysis = Analysis(self.data)

        # Use your scatter() in Analysis to handle the plotting of points. Note that it returns the (x, y) coordinates of the points.
        x, y = analysis.scatter(ind_var, dep_var, title)

        # Sample evenly spaced x values for the regression line between the min and max x data values
        x_reg = np.linspace(np.min(x), np.max(x), 100)

        # Use your regression slope, intercept, and x sample points to solve for the y values on the regression line.
        y_reg = self.slope * x_reg + np.full_like(x_reg, self.intercept)

        # Plot the line on top of the scatterplot.
        plt.plot(x_reg, y_reg.flatten(), color='red')

        # Calculate the R^2 value
        r_squared = self.r_squared(self.predict())

        # Make sure that your plot has a title (with R^2 value in it)
        plt.title(title + ' R^2 = ' + str(r_squared))

        plt.show()

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''

        from analysis import Analysis

        # Creating an analysis object
        analysis = Analysis(self.data)

        # Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        # Note that this method returns the figure and axes array that you will need to superimpose
        # the regression lines on each subplot panel.
        fig, ax = analysis.pair_plot(data_vars, fig_sz, title = 'Pair Plot')

        # Looping through the data variables
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                # Getting the independent and dependent variables
                ind_var = data_vars[i]
                dep_var = data_vars[j]

                # Getting the x and y values
                x = self.data.select_data([ind_var]).flatten()
                y = self.data.select_data([dep_var]).flatten()

                # Sample evenly spaced x values for the regression line between the min and max x data values
                x_reg = np.linspace(np.min(x), np.max(x), 100)

                # Calling the linear regression method to get the slope and intercept
                self.linear_regression([ind_var], dep_var)

                # Use your regression slope, intercept, and x sample points to solve for the y values on the regression line.
                y_reg = self.slope * x_reg + np.full_like(x_reg, self.intercept)

                # Check if hists_on_diag is true
                if hists_on_diag and i == j:
                    numVars = len(data_vars)
                    ax[i, j].remove()
                    ax[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
                    if j < numVars-1:
                        ax[i, j].set_xticks([])
                    else:
                        ax[i, j].set_xlabel(data_vars[i])
                    if i > 0:
                        ax[i, j].set_yticks([])
                    else:
                        ax[i, j].set_ylabel(data_vars[i])
                    # Plot the histogram on the diagonal
                    ax[i, j].hist(x, bins=20)
                else:
                    # Plot the line on top of the scatterplot subplot panel.
                    ax[i, j].plot(x_reg, y_reg.flatten(), color='red')

                # Calculate the R^2 value
                r_squared = self.r_squared(self.predict())

                # Make sure that your plot has a title (with R^2 value in it) make the title small so its fits above the plot
                ax[i, j].set_title(ind_var + ' vs ' + dep_var + ' R^2 = ' + str(r_squared), fontsize=8)

        plt.show()

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        # Initialize matrix to hold polynomial features of A
        X = np.zeros((A.shape[0], p))

        # Compute polynomial features
        for i in range(p):
            X[:,i] = A[:,0]**(i+1)

        return X

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        # Set instance variables for independent and dependent variables
        self.ind_vars = [ind_var]
        self.dep_var = dep_var

        # Extract the independent and dependent variable data from the data object
        A = self.data.select_data([ind_var])
        self.y = self.data.select_data([dep_var])

        # Make the polynomial matrix
        self.A = self.make_polynomial_matrix(A, p)
        
        # Set the instance variable for the polynomial regression degree
        self.p = p

        # Fit the regression and get the coefficients
        c, _, _, _ = scipy.linalg.lstsq(self.A, self.y)

        # Set the instance variables for the slope, intercept, and residuals
        self.slope = c[1:]
        self.intercept = c[0]
        # Converting the intercept to a float
        self.intercept = float(self.intercept)

        # Compute R^2 on the fit
        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred)

        # Compute the residuals
        self.residuals = self.compute_residuals(y_pred)

        # Setting the mse
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        pass

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        pass
