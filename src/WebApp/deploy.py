import os
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore, storage, auth
from flask import Flask, render_template, flash, redirect, url_for, session, request, send_file, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from datetime import datetime
import cv2
from keras.models import load_model
import numpy as np
import requests
import io
import sys
from kanren import Relation, facts, run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CNN.tumor_type import get_type
from ultralytics import YOLO
import SimpleITK as sitk
import tensorflow as tf
from skimage.restoration import denoise_tv_chambolle as anisotropic_diffusion

MASK_FOLDER = 'masks'
UPLOAD_FOLDER = 'uploaded_images'
FILTERED_IMAGES_FOLDER = 'filtered_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FILTERED_IMAGES_FOLDER'] = FILTERED_IMAGES_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER
app.secret_key = "m4xpl0it"

# Initialize Firebase
cred = credentials.Certificate("F:/Abdelrhman/Rickman-main/src/WebApp/instance/rickman-de167-firebase-adminsdk-99sn0-b2c780d3f0.json")
firebase_app = initialize_app(cred, options={'storageBucket': 'rickman-de167.appspot.com'})
firestore_db = firestore.client()

# Initialize Firebase Authentication
auth_instance = auth

def authenticate_user(email, password):
    firebase_api_key = "AIzaSyD6VFKiUOUWqKuVJDpWAuFn2xL56cdfmhU"
    auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_api_key}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    response = requests.post(auth_url, json=payload)

    if response.ok:
        user_data = response.json()
        user_id = user_data.get("localId")
        return user_id
    else:
        print(f"Authentication error: {response.text}")
        return None

epsilon = 1e-5
smooth = 1

# Define Tversky and related functions
def tversky(y_true, y_pred):
    y_true_pos = np.ndarray.flatten(y_true)
    y_pred_pos = np.ndarray.flatten(y_pred)
    true_pos = np.sum(y_true_pos * y_pred_pos)
    false_neg = np.sum(y_true_pos * (1-y_pred_pos))
    false_pos = np.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return np.power((1-pt_1), gamma)

# Load the segmentation model
model_seg = load_model("F:\Abdelrhman\Rickman\src\CNN\models\seg_model.h5", custom_objects={"focal_tversky": focal_tversky,
                                                       "tversky": tversky,
                                                       "tversky_loss": tversky_loss})

# Load YOLO model
try:
    b_model = YOLO('F:\Abdelrhman\Rickman\src\CNN\Yolo\Tumor seg(n).pt')
    if b_model is not None:
        print("Model loaded successfully.")
    else:
        print("Model is not properly loaded.")
except Exception as e:
    print(f"Error loading the model: {e}")

def preprocess_image_yolo(image_path):
       
    inputImage = sitk.ReadImage(image_path, sitk.sitkFloat32)
    image = inputImage

    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image, maskImage)

    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

    img = sitk.GetArrayFromImage(corrected_image)
    
    im = img.astype(np.uint8)
    
    img_resized = cv2.resize(im, (256, 256))

    img = img_resized / 255

    gamma = 2.2

    gamma_corrected = np.array(255* img ** gamma, dtype = 'uint8')

    enhanced = cv2.equalizeHist(gamma_corrected)

    denoised = anisotropic_diffusion(enhanced)
    
    depth_reduction = denoised.astype(np.uint8)
    
    converted = cv2.cvtColor(depth_reduction, cv2.COLOR_GRAY2RGB)
    
    result = b_model([converted])
    for res in result:
        if res.masks is not None:
            multiple = tf.zeros((256, 256), dtype=tf.uint8)
            for mask in res.masks.data: 
                if len(res.masks.data) > 1:
                    multiple = tf.bitwise.bitwise_or(mask, multiple)
                    mask = multiple.numpy() * 255
                else:
                    mask = mask.numpy() * 255
        else:
            mask = np.zeros((256, 256), dtype=np.uint8)
            
        bmask_image_path = os.path.join(app.config['MASK_FOLDER'], 'bmask_' + os.path.basename(image_path))
        cv2.imwrite(bmask_image_path, mask)
            
    return bmask_image_path

def apply_image_processing(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img, (256, 256))
        img_standardized = np.array(img_resized, dtype=np.float64)
        img_standardized -= img_standardized.mean()
        img_standardized /= img_standardized.std()

        X = np.empty((1, 256, 256, 3))
        X[0,] = img_standardized
        predict = model_seg.predict(X)
        predicted_mask = predict.squeeze().round()

        img_with_overlay = img_resized.copy()
        img_with_overlay[predicted_mask == 1] = (0, 255, 150)

        # Convert predicted mask to 8-bit image
        predicted_mask = (predicted_mask * 255).astype(np.uint8)

        # Save the predicted image
        filtered_image_path = os.path.join(app.config['FILTERED_IMAGES_FOLDER'], 'filtered_' + os.path.basename(image_path))
        tmask_image_path = os.path.join(app.config['MASK_FOLDER'], 'mask_' + os.path.basename(image_path))
        cv2.imwrite(filtered_image_path, img_with_overlay)
        cv2.imwrite(tmask_image_path, predicted_mask)

        return filtered_image_path, tmask_image_path

    except Exception as e:
        print("Exception in image processing\n")
        print(e, '\n')
        return None

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_data(email):
    user_ref = firestore_db.collection('users').document(email)
    user_data = user_ref.get().to_dict()
    return user_data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/index_auth", methods=["GET", "POST"])
def index_auth():
    if 'user_id' in session:
        user_id = session['user_id']
        user_ref = firestore_db.collection('users').document(user_id)
        user_data = user_ref.get().to_dict()

        return render_template("index_auth.html", user_data=user_data)
    else:
        return redirect(url_for('login'))

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' in session:
        user_id = session['user_id']

        first_name = request.form['first_name']
        middle_name = request.form['middle_name']
        last_name = request.form['last_name']
        gender = request.form['gender']
        age = request.form['age']
        national_id = request.form['national_id']
        phone = request.form['phone']

        user_ref = firestore_db.collection('users').document(user_id)
        user_data = {
            'first_name': first_name,
            'middle_name': middle_name,
            'last_name': last_name,
            'gender': gender,
            'age': age,
            'national_id': national_id,
            'phone': phone
        }

        # Check if a profile picture file is uploaded
        if 'file' in request.files:
            f = request.files['file']
            if f and is_allowed_file(f.filename):
                filename = secure_filename(f.filename)
                uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(uploaded_image_path)

                # Upload the image to Firebase Storage
                bucket = storage.bucket(app=firebase_app)
                blob = bucket.blob(f'profile_images/{user_id}/{filename}')
                blob.upload_from_filename(uploaded_image_path)

                # Update the user's profile picture URL in Firestore
                blob.make_public()
                profile_picture_url = blob.public_url
                user_data['profile_picture'] = profile_picture_url

        # Update user data in Firestore
        user_ref.update(user_data)

    return redirect(url_for('index_auth'))

@app.route('/remove_profile_picture', methods=['POST'])
def remove_profile_picture():
    if 'user_id' in session:
        user_id = session['user_id']
        # Update the user's profile picture URL to an empty string in Firestore
        user_ref = firestore_db.collection('users').document(user_id)
        user_ref.update({'profile_picture': ''})
        return jsonify({'success': True})
    return jsonify({'success': False})

def delete_profile_picture(blob_path):
    bucket = storage.bucket(app=firebase_app)
    blobs = bucket.list_blobs(prefix=blob_path)
    for blob in blobs:
        blob.delete()

@app.route("/check_auth", methods=["GET"])
def check_auth():
    if 'user_token' in session:
        return jsonify({"authenticated": True})
    else:
        return jsonify({"authenticated": False})

@app.route("/instruct")
def instruct():
    return render_template("instructions.html")

@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f_name)
    paths = apply_image_processing(original_image_path)
    filtered_image_path = paths[0]
    tmask_image_path = paths[1]
    bmask_image_path = preprocess_image_yolo(original_image_path)

    session['tmask_image_path'] = tmask_image_path
    session['bmask_image_path'] = bmask_image_path

    grade = session.get('grade', None)
    tumor_type = session.get('type', None)
    
    user_data = get_user_data(session['user_id'])

    if grade and tumor_type:
        show_pip = False
    else:
        show_pip = pred != 'The image does not contain a brain tumor.'
    
    return render_template('pred.html', pred=pred, f_name=f_name, original_image_path=original_image_path, filtered_image_path=filtered_image_path, user_data=user_data, show_pip=show_pip, grade=grade, tumor_type=tumor_type)


@app.route("/t_type", methods=['POST', 'GET'])
def answers():
   
    s = request.form.get('smoking')
    hist = request.form.get('family_history')
    w = request.form.get('weight')
    exp = request.form.get('exposure')
    a = request.form.get('alcohol')
    app = request.form.get('past_tumor')
    o = request.form.get('other_cancer')
    strs = request.form.get('stress')
    g = request.form.get('sex')
    age = request.form.get('age')
    e = request.form.get('ethnicity')
    
    bmask_image_path = session.get('bmask_image_path')
    tmask_image_path = session.get('tmask_image_path')
    
    # Get the grade and type using the get_type function
    tumor_type, grade = get_type(bmask_image_path, tmask_image_path, g, s, age, e, hist, w, exp, a, app, o, strs)
    
    # Store grade and type in the session
    session['grade'] = grade
    session['type'] = tumor_type
    
    return redirect(url_for('pred_page'))
    

@app.route("/upload", methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            f = request.files['bt_image']
            filename = secure_filename(f.filename)

            if filename != '' and is_allowed_file(filename):
                uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(uploaded_image_path)

                bucket = storage.bucket(app=firebase_app)
                blob = bucket.blob(filename)
                blob.upload_from_filename(uploaded_image_path)

                with open(uploaded_image_path, "rb") as img:
                    predicted = requests.post("http://localhost:5000/predict", files={"file": img})
                    try:
                        predicted.raise_for_status()
                        prediction_data = predicted.json()
                        session['pred_label'] = prediction_data.get('result', '')
                        session['filename'] = filename
                        session['grade'] = None
                        session['type'] = None
                        return redirect(url_for('pred_page'))
                    except requests.exceptions.HTTPError as err:
                        flash(f'Error in prediction: {err}', 'error')

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template("upload.html")

@app.route("/get_uploaded_image", methods=['GET'])
def get_uploaded_image():
    try:
        filename = session.get('filename', None)
        if filename:
            bucket = storage.bucket(app=firebase_app)
            blob = bucket.blob(filename)
            image_data = blob.download_as_bytes()

            temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(temp_image_path, 'wb') as temp_image_file:
                temp_image_file.write(image_data)

            return send_file(temp_image_path, mimetype='image/jpeg') 

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return "Image not found"

@app.route("/get_filtered_image", methods=['GET'])
def get_filtered_image():
    try:
        filename = session.get('filename', None)
        if filename:
            filtered_image_path = os.path.join(app.config['FILTERED_IMAGES_FOLDER'], 'filtered_' + filename)
            return send_file(filtered_image_path, mimetype='image/jpeg')

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return "Filtered Image not found"

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('user_id', None)
    session.pop('user_token', None)
    return redirect(url_for('index'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        passw = request.form["passw"]

        token = authenticate_user(email, passw)

        if token:
            session['user_id'] = email
            session['user_token'] = token
            return redirect(url_for("index_auth"))
        else:
            flash('Invalid email or password', 'error')
    return render_template("login.html")

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        try:
            feedback_data = request.get_json()
            feedback_text = feedback_data.get('feedback')
            rating = feedback_data.get('rating')
            username = feedback_data.get('username')
            time_submitted = datetime.now()

            # Store feedback data in Firebase Firestore
            feedback_ref = firestore_db.collection('feedback')
            feedback_ref.add({
                'text': feedback_text,
                'rating': rating,
                'username': username,
                'time_submitted': time_submitted
            })

            return jsonify({'success': True})
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': False})

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fname = request.form['fname']
        mname = request.form['mname']
        lname = request.form['lname']
        gender = request.form.get('gender', '')
        age = request.form['age']
        national_id = request.form['national_id']
        mail = request.form['mail']
        phone = request.form['phone']
        passw = request.form['passw']
        confirm_passw = request.form['confirm_passw']

        if passw != confirm_passw:
            return render_template("register.html", error="Passwords do not match")

        try:
            user = auth_instance.create_user(
                email=mail,
                email_verified=True,
                password=passw,
                display_name=f"{fname} {lname}",
            )

            user_data = {
                'first_name': fname,
                'middle_name': mname,
                'last_name': lname,
                'gender': gender,
                'age': age,
                'national_id': national_id,
                'email': mail,
                'phone': phone,
                'password': passw
            }
            
            user_ref = firestore_db.collection('users').document(mail)
            user_ref.set(user_data)

            flash('Account created successfully!', 'success')
            return redirect(url_for("login"))
        except Exception as e:
            print(f"Authentication error: {e}")
            return render_template("register.html", error="Error creating account")


    return render_template("register.html")

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form["email"]
        
        flash('Password reset link sent to your email', 'success')
    return render_template("forgot_password.html")

if __name__ == "__main__":
    app.run(debug=True, port=3000)
    
#cd F:\Abdelrhman\Rickman\src\WebApp
#python deploy.py