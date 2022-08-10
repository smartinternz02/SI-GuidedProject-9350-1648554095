from flask import Flask,request,render_template
import numpy as np
import joblib
app=Flask(__name__)
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "L7lwfnsD3-EcUf2o1Wys4CWpSltZ7a9adnqEeTzSs6ZO"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
model=joblib.load('../bagging.model')
sc=joblib.load('../transform.save')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/home',methods=['POST',"GET"])
def my_home():
    return render_template('home.html')
@app.route('/predict',methods=["POST","GET"])
def predict():
    input_feature=[float(x) for x in request.form.values()]
    x=[np.array(input_feature)]
    x=sc.transform(x)
    print(x)
    prediction=model.predict(x)
    labels=['Dark Trap','Emo','Hiphop','Pop','Rap','Rnb','Trap Metal','Underground Rap',\
           'dnb','hardstayle','psytrance','techhouse','techno','trance','trap']
    print("prediction is :",labels[prediction[0]])
    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": ['Dark Trap','Emo','Hiphop','Pop','Rap','Rnb','Trap Metal','Underground Rap',\
           'dnb','hardstayle','psytrance','techhouse','techno','trance','trap'],
                                       "values": [input_feature]}]}

    response_scoring = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/84e5a6eb-9385-401a-b8ea-8e4cbe3d3b5d/predictions?version=2022-08-090',
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    return render_template("result.html",prediction=labels[prediction[0]])
if __name__=="__main__":

    app.run(host='0.0.0.0', port=80000,debug=True)