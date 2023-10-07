import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('display.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    '''features = [[int(x) for x in request.form.values()]]
    pred = model.predict(features)''' 
    
    if request.method == 'POST':
      i1 = int( request.form['year'])
      i2 = int(request.form['present price'])
      i3 = int(request.form['kms driven'])
      i4 = int( request.form['fuel type'])
      i5 = int(request.form['seller type'])
      i6 = int(request.form['transmission'])
    inputs=[[i1,i2,i3,i4,i5,i6]]
    pred = model.predict(inputs)
    result=round(pred[0])

    return render_template('display.html', prediction='CAR PRICE IS : {}'.format(result))



if __name__ == "__main__":
    app.run(debug=True)