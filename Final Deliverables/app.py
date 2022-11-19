import numpy as np
import sklearn
from flask import Flask, request, jsonify, render_template
import pickle

import inputScript

app = Flask(__name__)
'''model = pickle.load(open('Phishing_Website.pkl','rb'))'''

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "caKSRo8qoQsPFfsf0LN4kPhWtWmBu2JFNBINokfQAjGY"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

@app.route('/')
def gui():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    return render_template('Final.html')


@app.route('/y_predict', methods=['POST',"GET"])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    url = request.form['url']
    checkprediction = inputScript.main(url)
    #prediction = model.predict(checkprediction)
    #print(prediction)
    payload_scoring = {"input_data": [{"fields": [ 'having_IPhaving_IP_Address', 'URLURL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report'],
                                       "values": checkprediction}]}

    response_scoring = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/66a39a69-b265-459f-a698-00c9bb92c031/predictions?version=2022-11-18',
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predict_val=response_scoring.json()
    print(predict_val)
    output = predict_val['predictions'][0]['values'][0][0]
    #output = 0
    if (output == 1):
        pred = "Your are safe!! This is a Legistimate Website."

    else:
        pred = "You are on the wrong site. Be cautious!"
    return render_template('Final.html', prediction_text='{}'.format(pred), url=url)


@app.route('/predict_api', methods=['POST','GET'])
def predict_api():
    '''For direct API calls trought request
    '''
    data = request.get_json(force=True)
    #prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=2000)


