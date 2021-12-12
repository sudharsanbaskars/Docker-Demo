from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        abs_path = str(request.form['filepath'])

        pred_val = pred_validation(abs_path) #object initialization

        pred_val.prediction_validation() #calling the prediction_validation function

        pred = prediction(abs_path) #object initialization

        # predicting for dataset present in database
        codn = pred.predictionFromModel()
        final_path = abs_path+"/Prediction_OutputFile/Predictions.csv"
        if codn is False:
            return Response("Please mention a valid path which contains valid files")
        else:
            return Response("Your Prediction File created at: "  +str(final_path))

    except Exception as e:
        return Response("Error Occurred! %s" %e)


@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRouteClient():
    try:
        folder_path = "Training_Batch_Files/"
        # if request.json['folderPath'] is not None:
        if folder_path is not None:
            path = folder_path
            path = request.json['folderPath']

            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function


            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successful!!")


port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 8000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

