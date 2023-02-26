'''data.py
Reads CSV files, stores data, access/filter data by variable name
CHANDRACHUD MALALI GOWDA
CS 251 Data Analysis and Visualization
Spring 2023
'''

import numpy as np

class Data:

    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''

        # Declaring the instance variables
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col

        # Checking if the filepath is not None
        if filepath is not None:
            self.read(filepath)

        pass

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''

        # Importing the csv module
        import csv

        # Opening the file in read mode
        with open(filepath, 'r') as file:
            # Reading the file
            reader = csv.reader(file)
            # Getting the headers of the file stripping the whitespaces
            self.headers = [header.strip() for header in next(reader)]
            # Storing the data in a list
            self.data = list(reader)

        # List to store the columns to delete
        todelete = []

        # Throw error when reading csv files without type specification in the first row using the isAlpha function
        if not self.data[0][0].isalpha():
            raise ValueError("The first row of the csv file should contain the type of the data")

        # Deleting the columns that are not numeric by looking at the first row of data
        for i in range(len(self.data[0])):
            if self.data[0][i].strip() != 'numeric':
                todelete.append(i)

        # Deleting the columns taking into account index switching
        for i in range(len(todelete)):
            for row in self.data:
                del row[todelete[i] - i]

        # Deleting the headers of the columns to delete
        for i in range(len(todelete)):
            del self.headers[todelete[i] - i]


        # Storing the headers in a dictionary stripping the whitespaces
        self.header2col = {header.strip(): i for i, header in enumerate(self.headers)}

        # Skipping the first row of data
        self.data = self.data[1:]

        # Converting the data to float
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                # Check that the data is not empty
                if self.data[i][j] != '':
                    self.data[i][j] = float(self.data[i][j])
                else:
                    self.data[i][j] = 0.0

        
        # Converting the data to numpy array
        self.data = np.array(self.data)

        pass

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''

        # Prints output in this format:
        #         data/iris.csv (150x4)
        # Headers:
        #   sepal_length    sepal_width    petal_length    petal_width
        # -------------------------------
        # Showing first 5/150 rows.
        # 5.1    3.5    1.4    0.2
        # 4.9    3.0    1.4    0.2
        # 4.7    3.2    1.3    0.2
        # 4.6    3.1    1.5    0.2
        # 5.0    3.6    1.4    0.2

        # Declaring the output string
        output = ''

        # Adding the file path and the dimensions of the data
        output += self.filepath + ' (' + str(len(self.data)) + 'x' + str(len(self.headers)) + ')\n'
        # Adding the headers
        output += 'Headers:\n'
        for header in self.headers:
            output += '  ' + header + '\t'
        output += '\n'
        # Adding the separator
        output += '-----------------------\n'
        # Adding the number of rows to show if the rows >= 5
        if len(self.data) >= 5:
            output += 'Showing first 5/' + str(len(self.data)) + ' rows.\n'
        # Adding the data
        for i in range(min(len(self.data), 5)):
            for j in range(len(self.data[i])):
                output += str(self.data[i][j]) + '\t'
            output += '\n'

        # Returning the output
        return output

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''

        # Returns a list of headers stripped of whitespaces
        return [header.strip() for header in self.headers]
    

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''

        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''

        return len(self.headers)

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''

        return len(self.data)

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''

        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''

        # Returns a list of the indices of the headers in the list headers
        return [self.header2col[header] for header in headers]

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''

        import numpy as np
        return np.copy(self.data)

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''

        return self.data[:5]

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''

        return self.data[-5:]

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''

        self.data = self.data[start_row:end_row]

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''

        # Getting the indices of the headers
        indices = self.get_header_indices(headers)
        
        # If rows is empty, return all samples
        if len(rows) == 0:
            return self.data[:, indices]
                
        # Otherwise, return ndarray the samples at the indices specified by the rows list
        
        # Declaring the output ndarray
        output = np.zeros((len(rows), len(headers)))

        # Filling the output ndarray
        for i in range(len(rows)):
            for j in range(len(headers)):
                output[i][j] = self.data[rows[i]][indices[j]]

        # Returning the output ndarray
        return output
