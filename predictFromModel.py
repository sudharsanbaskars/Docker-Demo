import pandas as pd
import os
import json
import shutil
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation


class prediction:

    def __init__(self,path):
        self.abs_path = path
        self.file_object = "Prediction_Log.txt"
        self.log_writer = logger.App_Logger()
        if path is not None:
            self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            #code change
            wafer_names=data['Wafer']
            #data=data.drop('Wafer',axis="columns")

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            is_null_present=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            # Do standardization
            data = preprocessor.standardization_for_test_data(data.drop('Wafer', axis="columns"))
            data['Wafer'] = wafer_names
            # Perform principle component analysis
            # with open("pca_components.json", "r") as f:
            #     dt = json.load(f)
            #     f.close()
            # n_components = dt["n_components"]
            #
            # data = preprocessor.pca_for_test_data(n_components, data)
            # data = pd.DataFrame(data)
            # data["Wafer"] = wafer_names

            #cols_to_drop=preprocessor.get_columns_with_zero_std_deviation(data)
            #data=preprocessor.remove_columns(data,cols_to_drop)
            #data=data.to_numpy()

            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            ##Code changed
            #pred_data = data.drop(['Wafer'],axis=1)
            clusters=kmeans.predict(data.drop(['Wafer'],axis=1))#drops the first column for cluster prediction
            data['clusters']=clusters
            clusters=data['clusters'].unique()
            print(clusters.shape)
            for i in clusters:
                cluster_data= data[data['clusters']==i]
                wafer_names = list(cluster_data['Wafer'])
                cluster_data=data.drop(labels=['Wafer'],axis=1)
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                result=list(model.predict(cluster_data))
                result = pd.DataFrame(list(zip(wafer_names,result)),columns=['Wafer','Prediction'])

                final_folder_path = self.abs_path + "/Prediction_Output_File"
                final_file_path = final_folder_path+"/Predictions.csv"

                if os.path.exists(final_folder_path) == False:
                    os.mkdir(final_folder_path)
                # else:
                #     shutil.rmtree(final_folder_path)
                #     os.mkdir(final_folder_path)

                result.to_csv(final_file_path,header=True,mode='a+') #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
            return True

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!!' + str(ex))
            return False






