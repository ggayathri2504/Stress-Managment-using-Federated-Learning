from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
app = Flask(__name__,template_folder='templates')

# model = tf.saved_model.load(r'C:\Users\HP\python_vscode\FL\my_model')
@app.route('/',methods=['GET','POST'])
def welcome():
        return render_template('welcome.html')
# Define your prediction function here
def make_prediction(input_data):
    # Code to make a prediction with the input data
    # model = tf.saved_model.load(r'C:\Users\HP\python_vscode\FL\my_model')
    model_s = load_model('model.h5')
    global_weights = model_s.get_weights()
    model=Sequential.from_config(model_s.get_config())
    model.set_weights(global_weights)
    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    prediction = model.predict(np.array(input_data)).argmax(axis=-1)
    client_weights=model.get_weights()
    # global_weights=[(global_weights+client_weights)/2]
    weigh=[]
    for i in range(len(global_weights)):
        weigh.append((client_weights[i]+global_weights[i])/2)
    global_weights=weigh
    model_s.set_weights(global_weights)
    model.save('model.h5')
    return prediction[0]

# sr-snoring range,rr-respiration rate,t-body temperature,lm-limb movement
# bo-blood oxygen,rem-eye movement,sr.1- sleep hours,hr-heart rate,sl-stress level
@app.route('/input', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = []
        input_data.append(float(request.form['sr']))
        input_data.append(float(request.form['rr']))
        input_data.append(float(request.form['t']))
        input_data.append(float(request.form['lm']))
        input_data.append(float(request.form['bo']))
        input_data.append(float(request.form['rem']))
        input_data.append(float(request.form['sr1']))
        input_data.append(float(request.form['hr']))
        input_data = np.array(input_data).reshape(1, -1)
        prediction = make_prediction(input_data)
        return render_template('result.html', prediction=prediction)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
