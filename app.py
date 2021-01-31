from flask import Flask, render_template, request
import numpy as np
from sklearn import tree
import pickle

with open('model/final_prediction.pickle', 'rb') as file:  
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data1 = request.form['Data1']
        data2 = request.form['Data2']
        data = [[data1, data2]]
        pred = model.predict(data)
        return render_template('index.html', pred=str(pred))
    return render_template('index.html')

if __name__ == '__main__':
    app.run()