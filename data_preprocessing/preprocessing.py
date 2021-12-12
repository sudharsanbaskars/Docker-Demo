import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.impute import KNNImputer
class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.
        Version: 1.0
        Revisions: None
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def standardization(self, data):
        try:
            self.logger_object.log(self.file_object,
                                   'Entered the standardization method of the Preprocessor class')
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(data)
            path = "models/scaler.save"
            if os.path.exists(path):
                os.remove(path)
            joblib.dump(scaler, path)
            standardized_data = pd.DataFrame(standardized_data)
            return standardized_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                'Exception occured in standardization method of the Preprocessor class. Exception message:' + str(e))
            raise e

    def standardization_for_test_data(self, data):
        try:
            self.logger_object.log(self.file_object,
                                   'Entered the standardization for test data method of the Preprocessor class')

            path = "scaler/scaler.save"
            scaler = joblib.load(path)
            standardized_data = scaler.fit_transform(data)
            standardized_data = pd.DataFrame(standardized_data)
            return standardized_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                'Exception occured in std for test data method of the Preprocessor class. Exception message:' + str(e))
            raise e

    def implement_pca(self,data):
        try:
            self.logger_object.log(self.file_object,
                                   'Entered the implement_pca method of the Preprocessor class')
            pca = decomposition.PCA(0.95)
            pca_data = pca.fit_transform(data)
            return pca.n_components_, pca_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                    'Exception occured in implement method of the Preprocessor class. Exception message:' + str(e))
            raise e

    def pca_for_test_data(self, n_components, data):
        try:
            self.logger_object.log(self.file_object,
                                   'Entered the pca_for_test method of the Preprocessor class')
            pca = decomposition.PCA(n_components=n_components)
            pca_data = pca.fit_transform(data)
            return pca_data

        except Exception as e:
            self.logger_object.log(self.file_object,
                'Exception occured in pca for test data method of the Preprocessor class. Exception message:' + str(
                                       e))
            raise e



    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception
                Version: 1.0
                Revisions: None
        """

        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method separates the features and a Label Coulmns.
            Output: Returns two separate Dataframes, one containing features and the other containing Labels .
            On Failure: Raise Exception
            Version: 1.0
            Revisions: None
        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self,data):
        """
                Method Name: is_null_present
                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                On Failure: Raise Exception
                Version: 1.0
                Revisions: None
        """

        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            if(self.null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data):
        """
                        Method Name: impute_missing_values
                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                        Output: A Dataframe which has all the missing values imputed.
                        On Failure: Raise Exception
                        Version: 1.0
                        Revisions: None
     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data=pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()


    def get_columns_with_zero_std_deviation(self,data):
        """
                    Method Name: get_columns_with_zero_std_deviation
                    Description: This method finds out the columns which have a standard deviation of zero.
                    Output: List of the columns with standard deviation of zero
                    On Failure: Raise Exception

                    Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None
        """
        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            print(data.shape)
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()