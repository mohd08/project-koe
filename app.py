import pandas as pd
import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow import keras
from tensorflow.keras.models import load_model
model = load_model("prediction.h5")
import tensorflow.compat.v1 as tf
global graph
graph = tf.get_default_graph

app = Flask(__name__)

picFolder = os.path.join('static', 'pics')
print(picFolder)
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/', methods=['GET', 'POST'])
def index():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'Logo_IIUM.png')
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])

def data():
    if request.method == 'POST':
        file = request.form['upload-file']
        dataset = [np.array(file)]
        with graph.as_default():
            pred = model.predict(dataset)
        data = pd.read_excel(file)
        return render_template('data.html', data=data.to_html())


if __name__ == "__main__":
    app.run(debug=True)