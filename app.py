from flask import Flask,render_template,request,redirect,url_for,session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func

import os
import sklearn
import tensorflow as tf
from keras.layers import Layer, Reshape, Activation, Conv1D, Conv1DTranspose, Dropout
from keras.layers import Input, Add, Concatenate, Embedding,LeakyReLU,Dense, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import RandomNormal
import numpy as np

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
import numpy as np

from tensorflow.keras.models import load_model

# Load the model
model = load_model('./models/vgg16_model.h5')
print(model.summary())

# Use the model for prediction, evaluation, etc.
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'database.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = "static/uploads"

class DoctorUser(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100),nullable=True)
    specialised_in = db.Column(db.String(100),nullable =True)
    hospital = db.Column(db.String(100),nullable =True)
    images = db.Column(db.String(255))


from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
import os
from werkzeug.utils import secure_filename
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    diseases = db.relationship('Disease', back_populates='user', lazy=True)

    
    def __repr__(self):
        return f'<User {self.email},{self.password}>'

class Disease(db.Model):
    __tablename__ = 'diseases'

    id = db.Column(db.Integer, primary_key=True)
    patientId = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    patientName = db.Column(db.String(100),nullable=True)
    doctorId = db.Column(db.Integer, nullable=True)
    doctorName = db.Column(db.String(100), nullable=True)
    disease = db.Column(db.String(100), nullable=True)
    prescription = db.Column(db.String(100), nullable=True)
    mriImage = db.Column(db.String(255))
    appointmentData = db.Column(db.String(100), nullable=True)
    status = db.Column(db.Boolean, nullable=False, default=False)
    doctorprescription = db.Column(db.String(100), nullable=True)
    date = db.Column(db.String(100), nullable=True)
    overlayed = db.Column(db.String(255))
    user = db.relationship('User', back_populates='diseases')


    

    
def predictionList(predicted_class):
    if(predicted_class == "Bacterial Pneumonia"):
        data = {
            "name":"Bacterial Pneumonia",
            "prescription":"The first-line treatment for pneumonia in adults is macrolide antibiotics, like azithromycin or erythromycin"
        }
        return data
    elif(predicted_class == "Corona Virus Disease"):
        data = {
            "name":"Corona Virus Disease",
            "prescription":"Both Paxlovid and Lagevrio must be started within the first 5 days of symptom onset. A third antiviral treatment called Remdesivir is FDA approved for people diagnosed with COVID-19. This is an intravenous treatment that is available at some health care facilities."
        }
        return data
    elif(predicted_class == "Normal"):
        data = {
            "name":"Normal",
            "prescription":"Stay Healty"
        }
        return data
    elif(predicted_class == "Tuberculosis"):
        data = {
            "name":"Tuberculosis",
            "prescription":"The most common treatment for active TB is isoniazid INH in combination with three other drugs—rifampin, pyrazinamide and ethambutol"
        }
        return data
    elif(predicted_class == "Viral Pneumonia"):
        data = {
            "name":"Viral Pneumonia",
            "prescription":"Antiviral medications: Viral pneumonia usually isn't treated with medication and can go away on its own. A provider may prescribe antivirals such as oseltamivir (Tamiflu®"
        }
        return data

# @app.route('/',methods=['POST','GET'])
# def LoginSection():
#     if request.method=='POST':
#         email = request.form.get('email')
#         password = request.form.get('password')
#         if(email == "admin@gmail.com" and password=="admin"):
#             print("yes")
#             return render_template('admindashboard.html')
#         data = User.query.filter_by(email=email,password=password).all()
#         if data:
#             session['id'] = id
#             return redirect(url_for('HomeSection'))
#     return render_template('index.html')
    

@app.route('/', methods=['POST', 'GET'])
def landingPage():
    return render_template("landing.html")

@app.route('/login', methods=['POST', 'GET'])
def LoginSection():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == "admin@gmail.com" and password == "admin":
            return render_template('admindashboard.html')
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['id'] = user.id  # Assign the user's ID to the session
            return redirect(url_for('HomeSection'))
    return render_template('index.html')


@app.route('/register',methods=['POST','GET'])
def RegisterSection():
    if request.method=='POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        retypepassword = request.form.get('retypepassword')
        filterEmail = User.query.filter_by(email = email).all()
        if filterEmail:
            return render_template('register.html',status='Email Already Registered')
        elif password==retypepassword:
            data = User(
                       email= email, 
                       password=password,name=name)
            db.session.add(data)
            db.session.commit()
            return redirect(url_for('LoginSection'))
           

    return render_template('register.html',status='')


@app.route("/success",methods=["GET","POST"])
def successfullyApplied():
    return render_template("appliedforappointment.html")

@app.route('/home',methods=['POST','GET'])
def HomeSection():
    id = session["id"]
    total_approved_diseases = Disease.query.filter_by(status=True,patientId=id).count()
    return render_template("upload.html",data = total_approved_diseases)

@app.route('/notify',methods=['POST','GET'])
def Notification():
    id = session["id"]
    data = Disease.query.filter_by(status=True,patientId=id).all()
    print(data)
    return render_template("notification.html",data = data)

# @app.route('/upload',methods=['POST','GET'])
# def UploadImage():
#     file = request.files['fileInput']
#     id = session['id']
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)
#     if file:
#         img_path = 'path_to_save_uploaded_image.jpg'
#         file.save(img_path)
#         img = image.load_img(img_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)

#         # Make prediction
#         preds = model.predict(x)

#         # Interpret the prediction
#         label = np.argmax(preds)
#         print("Predicted class index:", label)
#         class_type = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

#         # Assuming `label` is the predicted class index
#         predicted_class = class_type[label]
#         print("Predicted class:", predicted_class)
#         result = predictionList(predicted_class)
#         data = Disease(patientId=id,
#                        disease= result['name'], 
#                        prescription=result['prescription'],
#                        mriImage = file)
#         db.session.add(data)
#         db.session.commit()
#     return render_template("upload.html")

# @app.route('/upload',methods=['POST','GET'])
# def UploadImage():
#     file = request.files['fileInput']
#     id = session['id']
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)
#     if file:
#         img = image.load_img(file_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)

#         # Make prediction using the VGG16 model
#         preds = model.predict(x)

#         # Interpret the prediction
#         label = np.argmax(preds)
#         print("Predicted class index:", label)
#         class_type = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

#         # Assuming `label` is the predicted class index
#         predicted_class = class_type[label]
#         print("Predicted class:", predicted_class)
#         result = predictionList(predicted_class)
#         data = Disease(patientId=id,
#                        disease=result['name'],
#                        prescription=result['prescription'],
#                        mriImage=file_path)
#         db.session.add(data)
#         db.session.commit()
#     return render_template("upload.html")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import numpy as np
import os
from flask import request, render_template, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # Define the gradient model using the provided model inputs and outputs
#     grad_model = Model(
#         inputs=model.inputs,
#         outputs=[model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     # Record operations for automatic differentiation
#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # Get the gradients of the predicted class with respect to the output feature map
#     grads = tape.gradient(class_channel, last_conv_layer_output)

#     # Pool the gradients over all the axes leaving out the channel dimension
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # Multiply each channel in the feature map array by "how important this channel is" with regard to the predicted class
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # Normalize the heatmap between 0 and 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[18:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purposes, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()
# @app.route('/upload', methods=['POST', 'GET'])
# def UploadImage():
#     file = request.files['fileInput']
#     id = session['id']
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     file.save(file_path)
#     if file_path:
#         img = image.load_img(file_path, target_size=(224, 224))
#         # x = image.img_to_array(img)
#         # x = np.expand_dims(x, axis=0)

#         img = img.resize((256, 256))
#         x = np.array(img)
#         x = np.expand_dims(x, axis=-1)  # Add a channel dimension
#         x = np.expand_dims(x, axis=0) 
#         x = preprocess_input(x)
#         # Load the model

#         # # Generate the heatmap
#         # last_conv_layer_name = 'vgg16'
#         # heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)

#         # # Resize the heatmap to match the original image size
#         # heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))

#         # # Apply colormap
#         # heatmap_resized = np.uint8(255 * heatmap_resized)
#         # heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

#         # # Add the heatmap to the original image
#         # superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap_colored, 0.4, 0)

#         # # Save the overlayed image to the static/uploads/ directory
#         # overlayed_img_path = os.path.join('static', 'uploads', 'overlayed_' + filename)
#         # cv2.imwrite(overlayed_img_path, superimposed_img)

#         # Make prediction using the VGG16 model
#         preds = model.predict(x)
#         label = np.argmax(preds)
#         class_type = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
#         predicted_class = class_type[label]
#         result = predictionList(predicted_class)
#         # Store the data in the database
#         user = User.query.filter_by(id=id).first()
#         data = Disease(patientId=id,
#                        patientName=user.name,
#                        disease=predicted_class,
#                        prescription=result['prescription'],
#                        mriImage=filename,
#                        overlayed='overlayed_' + filename)
#         db.session.add(data)
#         db.session.commit()

#         # Query the database for the updated data
#         data = Disease.query.filter_by(patientId=id).order_by(Disease.id.desc()).all()
#         return render_template("resultlisting.html", data=data)
#     return render_template("upload.html")
def model_predict(img_path, model):
    img = image.load_img(img_path, color_mode = 'grayscale', target_size=(256, 256))  # Resize image to 256x256 and convert to grayscale

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Scale the pixel values to [0, 1]
    x = x / 255.0

    preds = model.predict(x)
    return preds
@app.route('/upload', methods=['POST', 'GET'])
def UploadImage():
    file = request.files['fileInput']
    id = session.get('id')
    if not id:
        return "User ID not found in session"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    if file_path:
        # img = image.load_img(file_path, target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # img = image.load_img(file_path, target_size=(224, 224))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)

        # # Make prediction using the pre-trained model
        # preds = model.predict(x)
        # label = np.argmax(preds)
        # class_type = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        # predicted_class = class_type[label]
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model_predict(file_path, model)

        # Process your result for human
        class_labels = ['Bacterial Pneumonia','Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia'] # Replace with your actual labels
        pred_class = class_labels[np.argmax(preds)]
        # result = str(pred_class) 
        # Make prediction using the VGG16 model
        # preds = model.predict(x)
        # label = np.argmax(preds)
        # class_type = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        # predicted_class = class_type[label]
        result = predictionList(pred_class)

        # Store the data in the database
        user = User.query.filter_by(id=id).first()
        data = Disease(patientId=id,
                       patientName=user.name,
                       disease=pred_class,
                       prescription=result['prescription'],
                       mriImage=filename,
                       overlayed='overlayed_' + filename)
        db.session.add(data)
        db.session.commit()

        # Query the database for the updated data
        data = Disease.query.filter_by(patientId=id).order_by(Disease.id.desc()).all()
        return render_template("resultlisting.html", data=data)

    return render_template("upload.html")
# @app.route('/upload', methods=['POST', 'GET'])
# def UploadImage():
#     file = request.files['fileInput']
#     id = session['id']
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     file.save(file_path)
#     if file_path:
#         # img = image.load_img(file_path, target_size=(224, 224), color_mode="rgb")
#         # x = image.img_to_array(img)
#         # x = np.expand_dims(x, axis=0)
#         # x = preprocess_input(x)

#         # # Make prediction using the VGG16 model
#         # preds = model.predict(x)
#         # label = np.argmax(preds)
#         # img = image.load_img(file_path, target_size=(224, 224), color_mode="rgb")
#         img = image.load_img(file_path, target_size=(224, 224))
#         # x = image.img_to_array(img)
#         # x = np.expand_dims(x, axis=0)
#         # x = preprocess_input(x)
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)

#         # Make prediction using the VGG16 model
#         preds = model.predict(x)
#         label = decode_predictions(preds, top=3)[0]
#         # label = np.argmax(preds)

#         class_type = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
#         predicted_class = class_type[label]
#         result = predictionList(predicted_class)
#         # Store the data in the database
#         user = User.query.filter_by(id=id).first()
#         data = Disease(patientId=id,
#                        patientName=user.name,
#                        disease=predicted_class,
#                        prescription=result['prescription'],
#                        mriImage=filename,
#                        overlayed='overlayed_' + filename)
#         db.session.add(data)
#         db.session.commit()

#         # Query the database for the updated data
#         data = Disease.query.filter_by(patientId=id).order_by(Disease.id.desc()).all()
#         return render_template("resultlisting.html", data=data)
#     return render_template("upload.html")



@app.route("/detaileddisease/<int:id>" , methods = ['POST','GET'])
def detailedDisease(id):
    data = Disease.query.filter_by(id=id).first()
    doctor = DoctorUser.query.filter_by(specialised_in=data.disease).all()
    if request.method=='POST':
        doctors = request.form.get('doctor')
        d_data = DoctorUser.query.filter_by(id=doctors).first()
        date = request.form.get('date')
        data.date =date
        data.doctorId = d_data.id
        data.doctorName = d_data.name
        db.session.add(data)
        db.session.commit()
        return redirect(url_for('successfullyApplied'))
    return render_template("detaileddisease.html",data=data,doctor=doctor)

@app.route("/history" , methods = ['POST','GET'])
def HistoryDetailsList():
    id = session['id']
    data = Disease.query.filter_by(patientId=id).order_by(Disease.id.desc()).all()
    return render_template("historylist.html",data=data)

@app.route("/userdashboard" , methods = ['POST','GET'])
def userHome():
    return render_template("userdashboard.html")

@app.route("/doctorslist" , methods = ['POST','GET'])
def doctorsList():
    data = DoctorUser.query.all()
    return render_template("doctorslists.html",data=data)

@app.route("/admindashboard" , methods = ['POST','GET'])
def adminDashboard():
    doctorcount = DoctorUser.query.count()
    patientcount = User.query.count()
    data = DoctorUser.query.all()
    return render_template("admindashboard.html",doctorcount=doctorcount,patientcount=patientcount)

@app.route("/adddoctor" , methods = ['POST','GET'])
def addDoctor():
    if request.method=="POST":
        email = request.form.get('email')
        specialisedin = request.form.get('Specialised')
        password = request.form.get('password')
        name = request.form.get('name')
        hospital = request.form.get('hospital')
        image = request.files.get('image') 
        print(email,specialisedin,password,name,hospital,image)
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs('uploads', exist_ok=True)
        image.save(file_path)
        filterEmail = DoctorUser.query.filter_by(email = email).all()
        if filterEmail:
            return render_template('adddoctor.html',status='Email Already Registered')
        data = DoctorUser(email= email,password=password,specialised_in=specialisedin,name=name,images=image.filename,hospital=hospital)
        db.session.add(data)
        db.session.commit()
        return redirect(url_for('adminDashboard'))
    return render_template("adddoctor.html")

@app.route('/doctorlogin', methods=['POST', 'GET'])
def doctorLoginSection():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = DoctorUser.query.filter_by(email=email, password=password).first()
        if user:
            session['doctorid'] = user.id  # Assign the user's ID to the session
            return redirect(url_for('doctorHomePage'))
    return render_template('doctorlogin.html')

@app.route('/doctorhome',methods=["POST","GET"])
def doctorHomePage():
    doctorid = session['doctorid']
    user = DoctorUser.query.filter_by(id=doctorid).first()
    data = Disease.query.filter_by(doctorId=doctorid).all() 
    total_approved_diseases = Disease.query.filter_by(status=True).count()
    total_rejected_diseases = Disease.query.filter_by(status=False).count()
    return render_template("DoctorHomePage.html",user=user,data=data,total_rejected_diseases=total_rejected_diseases,total_approved_diseases=total_approved_diseases)



@app.route("/doctordetaileddisease/<int:id>" , methods = ['POST','GET'])
def detailedDoctorDisease(id):
    data = Disease.query.filter_by(id=id).first()
    doctorid = session['doctorid']
    user = DoctorUser.query.filter_by(id=doctorid).first()
    disease_history = Disease.query.filter_by(patientId=data.patientId,doctorId=doctorid).order_by(Disease.id.desc()).all()
    if request.method=='POST':
        status = request.form.get('status')
        doctorPrescription = request.form.get('doctorprescription')
        if(status=="0"):
            data.status=False
            data.doctorprescription =doctorPrescription
            db.session.add(data)
            db.session.commit()
            return redirect(url_for('doctorHomePage'))
        else:
            data.status=True
            data.doctorprescription =doctorPrescription
            db.session.add(data)
            db.session.commit()
            return redirect(url_for("doctorHomePage"))
    return render_template("doctordetailedview.html",data=data,user=user,disease=disease_history)


@app.route("/admindoctor",methods=["GET","POST"])
def adminDoctor():
    data = DoctorUser.query.all()
    return render_template("admindoctorlist.html",data=data)

@app.route("/doctorlogout" , methods = ['POST','GET'])
def doctorLogout():
    session.pop('doctorid', None)
    return redirect(url_for("doctorLoginSection"))

@app.route("/logout" , methods = ['POST','GET'])
def Logout():
    session.pop('id', None)
    return redirect(url_for("LoginSection"))



@app.route("/logouts" , methods = ['POST','GET'])
def AdminLogout():
    return redirect(url_for("LoginSection"))

if __name__ == "__main__":
    app.run(debug=True)
    
