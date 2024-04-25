from flask import Flask, render_template, request, redirect, url_for, flash, session, g
import pickle
from flask_cors import CORS
import numpy as np
import json
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2
import io, os
import base64
from mtcnn import MTCNN
from model import create_model
import pandas as pd
import glob
from functools import wraps
import sqlite3
from flask_sqlalchemy import SQLAlchemy
import h5py

DATABASE = 'deepfake.db'

custom_images = glob.glob('custom_data/*/*')

uploadedfakeimages = glob.glob('static/permittedimg/Fake/*')
uploadedfakeimages = [img.strip('static/') for img in uploadedfakeimages]
uploadedrealimages = glob.glob('static/permittedimg/Real/*')
uploadedrealimages = [img.strip('static/') for img in uploadedrealimages]

print(tf.__version__)

input_size = 224

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepfake.db'
app.secret_key = 'deepfake'
CORS(app)

dbconn = sqlite3.connect(DATABASE)
if dbconn is not None:
    print("Database connection established successfully")
dbconn.close()

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "static/custom_data/",
    target_size=(input_size, input_size),
    batch_size=1,
    class_mode=None,
    shuffle=True,
    classes = None,
    color_mode="rgb"
)


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:            
            print("You need to login first")
            #return redirect(url_for('login', error='You need to login first'))
            return render_template('logIn.html', error='You need to login first')    
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if ('logged_in' in session) and (session['usertype'] == 'admin'):
            return f(*args, **kwargs)
        else:            
            print("You need to login first as admin user")
            #return redirect(url_for('login', error='You need to login first as admin user'))
            return render_template('logIn.html', error='You need to login first as admin user')    
    return wrap

@app.route('/index')
@app.route('/home')
@app.route('/', methods = ['POST', 'GET'])
def index():
    if 'model' not in session:
        session['model'] = 'dfdc'

    return render_template('home.html')

@app.route('/index')
@app.route('/home')
@app.route('/', methods = ['POST', 'GET'])
def index_user():
    session['model'] = 'dfdc'
    return render_template('home.html', logged_in = session['logged_in'], usertype = session['usertype'])

@app.route('/predict/<name>', methods=['POST', 'GET'])
def predict(name):
    if request.method == 'POST':
        print("Predicting with model: ", session['model'])
        model = tf.keras.models.load_model('models/'+session['model']+'.h5')
        if name == 'uploads':
            data = request.files['image']
            image = Image.open(data)
            image.save('static/uploads/image.png')
            image = cv2.imread('static/uploads/image.png')
            detector = MTCNN()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(image)
            if len(faces) == 0:
                return render_template('error.html')
            else:
                return render_template('predict.html')
        elif name == 'examples1':
            image = cv2.imread('static/img/eg_1.jpeg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (input_size, input_size))
            image = np.array(image) / 255.0
            image = image.reshape(-1, input_size, input_size, 3)
            prediction = model.predict(image)
            if prediction[0][0] < 0.5:
                result = "Real"
                probability = round(prediction[0][0] * 100, 2)
            else:
                result = "Fake"
                probability = round(prediction[0][0] * 100, 2)
            return render_template('get_results.html', result = result, probability = probability, imagetype = 'example1')
        elif name == 'examples2':
            image = cv2.imread('static/img/eg_2.jpeg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (input_size, input_size))
            image = np.array(image) / 255.0
            image = image.reshape(-1, input_size, input_size, 3)
            prediction = model.predict(image)
            if prediction[0][0] < 0.5:
                result = "Real"
                probability = round(prediction[0][0] * 100, 2)
            else:
                result = "Fake"
                probability = round(prediction[0][0] * 100, 2)
            return render_template('get_results.html', result = result, probability = probability, imagetype = 'example2')
        elif name == 'examples3':
            image = cv2.imread('static/img/eg_3.jpeg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (input_size, input_size))
            image = np.array(image) / 255.0
            image = image.reshape(-1, input_size, input_size, 3)
            prediction = model.predict(image)
            if prediction[0][0] < 0.5:
                result = "Real"
                probability = round(prediction[0][0] * 100, 2)
            else:
                result = "Fake"
                probability = round(prediction[0][0] * 100, 2)
            return render_template('get_results.html', result = result, probability = probability, imagetype = 'example3')         
    return redirect(url_for('index'))

@app.route('/get_result/<name>', methods=['POST', 'GET'])
def result(name):
    if request.method == 'POST':
        
        if name == 'uploads':
            raw_image = cv2.imread('uploads/image.png')
        elif name == 'examples1':  
            raw_image = cv2.imread('static/img/eg_1.jpeg')
        elif name == 'examples2':
            raw_image = cv2.imread('static/img/eg_2.jpeg')
        elif name == 'examples3':
            image = cv2.imread('static/img/eg_3.jpeg')

           
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = np.array(image) / 255.0
        image = image.reshape(-1, input_size, input_size, 3)
        
        model = tf.keras.models.load_model('./models/'+session['model']+'.h5')
        print("Predicting with model: ", session['model'])
        model.summary()

        prediction = model.predict(image)
        if prediction[0][0] < 0.5:
            result = "Real"
            probability = round(prediction[0][0] * 100, 2)
        else:
            result = "Fake"
            probability = round(prediction[0][0] * 100, 2)
        
        if name == 'uploads':
            perm = request.form.get('saveimage')
            if perm != None:
                for i in range(10):
                    imgname = 'uploadedimage_' + str(i) + '.png'
                    if os.path.exists('static/permittedimg/Real/'+imgname) == True or os.path.exists('static/permittedimg/Fake'+imgname) == True:
                        imgname = 'uploadedimage' + str(i+1) + '.png'
                    else:
                        break
                print("Image is saved as: ", imgname)
                if result == 'Real':
                    cv2.imwrite('static/permittedimg/Real/'+imgname, raw_image)
                elif result == 'Fake':
                    cv2.imwrite('static/permittedimg/Fake/'+imgname, raw_image)
            else:
                print("Image is not saved.")
        return render_template('get_results.html', result = result, probability = probability, imagetype = 'upload')
    return redirect(url_for('index'))


@app.route('/model', methods=['POST', 'GET'])
@login_required
def models():
    if session['logged_in']:
        usertype = session['usertype']
        print(usertype)
    else:
        usertype = None
    summary = None
    message = None
    models = glob.glob('models/*')
    models = [os.path.basename(model) for model in models]
    if request.method == 'POST':
        new_model = request.files['modelfile']
        hf = h5py.File(new_model, 'r')
        
        placeholder = tf.keras.models.load_model(hf)
        placeholder.save("models/"+new_model.filename)
        new_model = tf.keras.models.load_model("models/"+new_model.filename)
        new_model.summary()
        session['model'] = new_model.filename.strip('.h5') 
    elif request.method == 'GET':
        message = request.args.get('message')
        modelname = request.args.get('modelname')
        if modelname is None:
            modelname = session['model']
        if message is not None:
            print(message)

    print("Model name: ", modelname)
    history = pd.read_pickle('history/'+modelname+'_history.pkl')
    train_acc = sum(history['accuracy']) / len(history['accuracy'])
    val_acc = sum(history['val_accuracy']) / len(history['val_accuracy'])
    train_loss = sum(history['loss']) / len(history['loss'])
    val_loss = sum(history['val_loss']) / len(history['val_loss'])
        
    return render_template('model.html', train_acc = train_acc, val_acc = val_acc, train_loss = train_loss, val_loss = val_loss, modelname=modelname, custom_images = custom_images, uploadedfakeimages=uploadedfakeimages, uploadedrealimages=uploadedrealimages, usertype=usertype, models = models, message = message, summary = summary) 

@app.route('/change_model/<name>', methods=['POST', 'GET'])
def change_model(name):
    if request.method == 'GET':
        model = tf.keras.models.load_model('models/'+name)
        modelname = name.strip('.h5')
        session['model'] = modelname
        return redirect(url_for('models', message = 'Model changed successfully', modelname=modelname) )

@app.route('/login', methods=['POST', 'GET'])  
def logIn():
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']
            with sqlite3.connect(DATABASE) as dbconn:
                cur = dbconn.cursor()
                cur.execute('SELECT * FROM User WHERE email = ? AND password = ?', (email, password))
                user = cur.fetchone()
                if user is not None:
                    session['logged_in'] = True
                    session['username'] = user[1]
                    session['email'] = user[2]
                    session['usertype'] = user[4]
                    print(user[4])
                    return redirect(url_for('profile'))
                else:
                    return render_template('logIn.html', error='Invalid email or password')
        except Exception as e:
            print("Database connection error")
            print(str(e))
        finally:
            dbconn.close()
    return render_template('logIn.html')

@app.route('/signup', methods=['POST', 'GET'])
def signUp():
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            usertype = 'standard'
            with sqlite3.connect(DATABASE) as dbconn:
                cur = dbconn.cursor()
                cur.execute('SELECT * FROM User WHERE email = ? OR username = ?', (email, username))
                user = cur.fetchone()
                if user is not None:
                    return render_template('signUp.html', error='User already exists')
                else:
                    cur.execute('INSERT INTO User (username, email, password, usertype) VALUES (?, ?, ?, ?)', (username, email, password, usertype))
                    print("User added successfully")
                    dbconn.commit()
        except Exception as e:
            dbconn.rollback()
            print("Database connection error")
            print(str(e))
        finally:
            dbconn.close()
        return redirect(url_for('logIn'))
    return render_template('signUp.html')

@app.route('/logout')
@login_required
def logout():
    session['logged_in'] = False
    return redirect(url_for('index'))

@app.route('/profile', methods=['POST', 'GET'])
@login_required
def profile():
    return render_template('profile.html', username = session['username'], email = session['email'], usertype = session['usertype'])

@app.route('/train', methods=['POST', 'GET'])     
#@admin_required
def train():
    return render_template('train.html')
    
if __name__ == '__main__':
    app.run(debug=True, port=8080)

    