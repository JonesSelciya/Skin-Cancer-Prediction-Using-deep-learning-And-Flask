from flask import Flask,render_template,redirect,request,flash,session,url_for
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2



import mysql.connector


conn=mysql.connector.connect(host="localhost",user="root",password="root",autocommit=True)
mycursor=conn.cursor(dictionary=True,buffered=True)
mycursor.execute("create database if not exists skincancer")
mycursor.execute("use skincancer")
mycursor.execute("create table if not exists skin(id int primary key auto_increment,cname varchar(255),email varchar(30) unique,cpassword text)")

model = tf.keras.models.load_model('modelfile/efficientnet.h5')

app=Flask(__name__)
app.secret_key = 'super secret key'
UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = 'uploads'\
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv', 'mov', 'jpg', 'jpeg', 'png', 'gif'}


app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



@app.route('/')
def home():
    return render_template('index.html')




@app.route('/registration',methods =['GET', 'POST'])
def registration():
  if request.method == 'POST' and 'pass' in request.form and 'email' in request.form and 'hos' in request.form:
        name = request.form.get('pass')
        password=request.form.get('hos')
        mob = request.form.get('mob')
        email = request.form.get('email')
        mycursor.execute("SELECT * FROM skin WHERE email = '"+ email +"' ")
        account = mycursor.fetchone()
        if account:
            flash('You are already registered, please log in')
        else:
            
            mycursor.execute("insert into skin values(NULL,'"+ name +"','"+ email +"','"+ password +"')")
            # msg=flash('You have successfully registered !')
            return render_template("login.html")
        
  return render_template("register.html")

@app.route('/login',methods =['GET', 'POST'])
def login():
    if request.method == 'POST' and 'nm' in request.form and 'pass' in request.form:
        print('hello')
        email = request.form['nm']
        password = request.form['pass']
        
        mycursor.execute("SELECT * FROM skin WHERE email = '"+ email +"' AND cpassword = '"+ password +"'")
        account = mycursor.fetchone()
        print(account)
        if account:
            session['loggedin'] = True
            session['email'] = account['email']
            msg = flash('Logged in successfully !')
                
            return redirect(url_for('upload'))
        else:
            msg = flash('Incorrect username / password !')
            return render_template('login.html',msg=msg)
    return render_template('login.html')




@app.route('/upload')
def upload():
    return render_template('upload.html')





@app.route('/image', methods=['POST', 'GET'])
def image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('image.html', error='No file part')

        file = request.files['image']

        if file.filename == '':
            return render_template('image.html', error='No selected file')

        if file:
            # Save the uploaded file to the 'upload' folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load the image and perform necessary preprocessing
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(300, 300))
            a = tf.keras.preprocessing.image.img_to_array(img)
            a = np.expand_dims(a, axis=0)
            imag = np.vstack([a])
            
            # Make a prediction using the loaded model
            predict = model.predict(imag, batch_size=1)
            classes = np.argmax(predict)
            class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis','Candidiasis Ringworm Tinea','Dermatofibroma','Melanocytic nevus','Melanoma','Normal','Squamous carcinoma cell','Vascular lesion']

            text = class_names[classes]

            # Display the result on the image
            reta = cv2.imread(file_path)
            reta = cv2.resize(reta, (500, 700))
            cv2.putText(reta, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 3)

            # Save the result image
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_image_path, reta)

            # Render the result page with the result image and predicted lab

            return render_template('image.html',  predicted=text,image_file=file_path)

    return render_template('image.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']




# @app.route('/video', methods=['POST', 'GET'])
# def vedio():
#     print("hello1")
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return render_template('video.html', error='No file part')

#         file = request.files['file']

#         if file.filename == '':
#             return render_template('video.html', error='No selected file')
#         print('hello2')

#         if file and allowed_file(file.filename):
#             # Save the uploaded file to the 'upload' folder
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             print('hello3')

#             # Capture video frames
#             cap = cv2.VideoCapture(file_path)
#             frame_width = int(cap.get(3))
#             frame_height = int(cap.get(4))
#             out = cv2.VideoWriter(os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename),
#                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Perform necessary preprocessing on each frame
#                 frame = cv2.resize(frame, (300, 300))
#                 img = tf.keras.preprocessing.image.img_to_array(frame)
#                 img = np.expand_dims(img, axis=0)
#                 imag = np.vstack([img])
#                 print('hello4')
#                 # Make a prediction using the loaded model
#                 predict = model.predict(imag, batch_size=1)
#                 classes = np.argmax(predict)
#                 class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Candidiasis Ringworm Tinea',
#                                'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Normal', 'Squamous carcinoma cell',
#                                'Vascular lesion']

#                 text = class_names[classes]
#                 print(text)
#                 # Display the result on the frame
#                 cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 3)

#                 # Save the result frame
#                 out.write(frame)
#             # Release the video capture and writer objects
#             cap.release()
#             out.release()
#             print('5')
#             # Render the result page with the result video and predicted label
#             return render_template('video_output.html', predicted=text, video_file=file_path)

#     return render_template('video.html')





app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv', 'mov', 'jpg', 'jpeg', 'png', 'gif'}





@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('video.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('video.html', error='No selected file')

        if file and allowed_file(file.filename):
            # Save the uploaded file to the 'upload' folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Capture video frames
            cap = cv2.VideoCapture(file_path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv2.VideoWriter(os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename),
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform necessary preprocessing on each frame
                frame = cv2.resize(frame, (300, 300))
                img = tf.keras.preprocessing.image.img_to_array(frame)
                img = np.expand_dims(img, axis=0)
                imag = np.vstack([img])

                # Make a prediction using the loaded model
                predict = model.predict(imag, batch_size=1)
                classes = np.argmax(predict)
                class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Candidiasis Ringworm Tinea',
                               'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Normal', 'Squamous carcinoma cell',
                               'Vascular lesion']

                text = class_names[classes]
                print(text)

                # Display the result on the frame
                cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 3)

                # Save the result frame
                out.write(frame)

            # Release the video capture and writer objects
            cap.release()
            out.release()

            # Render the result page with the result video and predicted label
            return render_template('video_output.html', predicted=text)

    return render_template('video.html')




if __name__=="__main__":
    app.run(debug=True)
