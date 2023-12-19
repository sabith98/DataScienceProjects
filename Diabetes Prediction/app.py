from flask import Flask,request,render_template,redirect,url_for
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')
    # return redirect(url_for('predictdata'))

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return '<h1>Welcome to diabeted Prediction</h1>'
    else:
        data = request.get_json()

        data=CustomData(
            Pregnancies=int(data['Pregnancies']),
            Glucose=int(data['Glucose']),
            BloodPressure=int(data['BloodPressure']),
            SkinThickness=int(data['SkinThickness']),
            Insulin=int(data['Insulin']),
            BMI=float(data['BMI']),
            DiabetesPedigreeFunction=float(data['DiabetesPedigreeFunction']),
            Age=int(data['Age'])
        )

        pred_df=data.get_data_as_data_frame()
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        # return render_template('home.html',results=results[0])

        result_str = "Patient has diabetes" if results[0]==1.0 else "Patient doesn't has diabetes"

        return result_str
    

if __name__=="__main__":
    app.run(host="0.0.0.0")