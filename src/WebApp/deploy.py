import os
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore, storage, auth
from flask import Flask, render_template, flash, redirect, url_for, session, request, send_file,jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
import cv2
import numpy as np
import requests

UPLOAD_FOLDER = 'uploaded_images'
FILTERED_IMAGES_FOLDER = 'filtered_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FILTERED_IMAGES_FOLDER'] = FILTERED_IMAGES_FOLDER
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


def apply_image_processing(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise Exception(f"Error loading image: {image_path}")

        dim = (500, 590)
        img = cv2.resize(img, dim)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        enhanced_image = cv2.equalizeHist(gray)

        _, thresh = cv2.threshold(enhanced_image, 155, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        colormask = np.zeros(img.shape, dtype=np.uint8)
        colormask[thresh != 0] = np.array((0, 0, 255))
        tumor_highlight = cv2.addWeighted(img, 0.7, colormask, 0.1, 0)

        ret, markers = cv2.connectedComponents(thresh)
        marker_area = [np.sum(markers == m) for m in range(1, np.max(markers) + 1)]
        largest_component = np.argmax(marker_area) + 1
        tumor_mask = markers == largest_component
        tumor_out = img.copy()
        tumor_out[tumor_mask == False] = (0, 0, 0)

        averaging = cv2.blur(tumor_out, (11, 11))

        _, thresh_after_smoothing = cv2.threshold(averaging, 155, 255, cv2.THRESH_BINARY)

        filtered_image_path = os.path.join(app.config['FILTERED_IMAGES_FOLDER'], 'filtered_' + os.path.basename(image_path))
        cv2.imwrite(filtered_image_path, thresh_after_smoothing)

        return filtered_image_path

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
        email = request.form['email']

        user_ref = firestore_db.collection('users').document(user_id)
        user_data = {
            'first_name': first_name,
            'middle_name': middle_name,
            'last_name': last_name,
            'gender': gender,
            'age': age,
            'national_id': national_id,
            'phone': phone,
            'email': email
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

        # Check if the email is being updated
        if email != session['user_id']:
            try:
                # Update email in Firebase Authentication
                user = auth.update_user(user_id, email=email)
            except Exception as e:
                print(f"Error updating email in Firebase Authentication: {e}")

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
    filtered_image_path = apply_image_processing(original_image_path)

    user_data = get_user_data(session['user_id'])

    return render_template('pred.html', pred=pred, f_name=f_name, original_image_path=original_image_path, filtered_image_path=filtered_image_path, user_data=user_data)

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