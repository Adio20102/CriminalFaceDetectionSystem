from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import base64
import os
import tempfile


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)



class CustomVideoCapture:
    def __init__(self):
        self.video_capture = cv2.VideoCapture()
        self.frame_count = 0

    def open(self, video_path):
        self.video_capture.open(video_path)

    def read(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.frame_count += 1
        return ret, frame

    def release(self):
        self.video_capture.release()


class DetectedCriminals(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    crime_type = db.Column(db.String(100))
    age = db.Column(db.Integer)
    national_verification_number = db.Column(db.String(100))
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    w = db.Column(db.Integer)
    h = db.Column(db.Integer)
    # Store the actual image data
    face_image = db.Column(db.LargeBinary)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register_form')
def register_form():
    return render_template('register_form.html')


@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        age = request.form['age']
        name = request.form['name']
        crime_type = request.form['crime_type']
        photo_file = request.files['photo']
        date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d')
        gender = request.form['gender']
        national_verification_number = request.form['national_verification_number']

        if age and crime_type and photo_file:
            photo_data = photo_file.read()  # Read the binary data of the uploaded photo

            # Face detection using Haar cascade
            img_array = np.frombuffer(photo_data, dtype=np.uint8)  # Convert binary data to numpy array
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

             # Load the Haar cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces in the uploaded image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if not len(faces):  # No faces detected
                flash('No faces Detected In The Uploaded Image.', 'error')
                return render_template('register_form.html')
            else:
                name = name.strip()
                if name == "":
                    flash('Input Criminal Name Properly.', 'info')
                    return render_template('register_form.html')

                crime_type = crime_type.strip()
                if crime_type == "":
                    flash('Input Crime Type Properly.', 'info')
                    return render_template('register_form.html')

                # Query the database for existing faces with matching national verification number
                existing_face = DetectedCriminals.query.filter_by(national_verification_number=national_verification_number).first()

                if existing_face:
                    # Check if any detected faces overlap with the existing face
                    for (x, y, w, h) in faces:
                        if x < existing_face.x + existing_face.w and \
                           x + w > existing_face.x and \
                           y < existing_face.y + existing_face.h and \
                           y + h > existing_face.y:
                            # If overlapping, perform necessary actions (e.g., update details)
                            if existing_face.date_of_birth == date_of_birth.date() and existing_face.name == name and existing_face.gender == gender:
                                if int(age) >= int(existing_face.age):
                                    existing_face.crime_type = existing_face.crime_type + ',' + crime_type
                                    # To handle  duplicate crimes
                                    temp=existing_face.crime_type.split(',')
                                    unique_elements = []
                                    for tp in temp:
                                        if tp not in unique_elements:
                                            unique_elements.append(tp)
                                    existing_face.crime_type=','.join(unique_elements)
                                    existing_face.age = age
                                    db.session.commit()
                                    flash('Criminal is Already Registered. All Details Updated Except Previously Registered Photo.', 'info')
                                    return render_template('register_form.html')
                                else:
                                    flash(f'Criminal Having Same Face and National Verification Number Was Previously Registered With Age: {existing_face.age}.', 'info')
                                    return render_template('register_form.html')
                            else:
                                flash('Criminal With Same National Verification Number Already Exists.', 'info')
                                return render_template('register_form.html')
                    # If no overlapping detected faces found, display relevant message
                    flash('A Criminal Already With Different Face Exists With National Verification Number: {existing_face.national_verification_number}', 'info')
                    return render_template('register_form.html')
                
                
                # To check is the face uploaded image is same as that of a registered criminal in the database
                stored_criminal_faces_ = DetectedCriminals.query.all()
                if stored_criminal_faces_:
                    for stored_criminal_face in stored_criminal_faces_:
                        stored_face_x = stored_criminal_face.x
                        stored_face_y = stored_criminal_face.y
                        stored_face_w = stored_criminal_face.w
                        stored_face_h = stored_criminal_face.h
                        for (x, y, w, h) in faces:
                          if x < stored_face_x + stored_face_w and \
                             x + w > stored_face_x and \
                             y < stored_face_y + stored_face_h and \
                             y + h > stored_face_y:
                             flash('Criminal With Same Face Already Exists.', 'warning')
                             return render_template('register_form.html') 


                for (x, y, w, h) in faces:
                    cropped_face = img[y:y+h, x:x+w]
                    # Convert image to byte array before storing
                    
                    _, encoded_image = cv2.imencode('.jpg', cropped_face)
                    image_as_bytes = encoded_image.tobytes()

                    # To ensure no duplicate crime is Registered
                    temp_var=crime_type.split(',')
                    unique_crime=[]

                    for tp in temp_var:
                         if tp not in unique_crime:
                             unique_crime.append(tp)
                    crime_type=','.join(unique_crime)
            
                    new_face = DetectedCriminals(
                        name=name,
                        date_of_birth=date_of_birth,
                        age=age,
                        gender=gender,
                        crime_type=crime_type,
                        national_verification_number=national_verification_number,
                        x=int(x),
                        y=int(y),
                        w=int(w),
                        h=int(h),
                        face_image=image_as_bytes  # Store the image data
                    )
                    db.session.add(new_face)
                    db.session.commit()
                    flash('Criminal Registration Successful', 'success')
                return render_template('register_form.html')
        else:
            flash('Please Fill Out All The Fields', 'error')
            return redirect(url_for('register_form'))
    return redirect(url_for('register_form'))


@app.route('/img_inv')
def img_inv():
    return render_template('img_inv.html')


@app.route('/img_inv', methods=['POST'])
def investigate():
    criminal = []
    if request.method == 'POST':
        photo_file = request.files['image']

        if photo_file:
            photo_data = photo_file.read()  # Read the binary data of the uploaded photo

            nparr = np.frombuffer(photo_data, dtype=np.uint8)  # Convert binary data to numpy array
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

             # Load the Haar cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces in the uploaded image
            faces_locations_ = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


            if len(faces_locations_) == 0:
                flash('No Faces Were Detected In The Uploaded Image.', 'error')
            else:
                
                # Encode the uploaded image to base64 for displaying
                _, img_encoded = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                stored_faces_ = DetectedCriminals.query.all()
                if not stored_faces_:
                    flash('No Criminal Faces Stored In The Database.', 'error')
                    return redirect(url_for('img_inv'))
                for stored_face in stored_faces_:
                    stored_face_x = stored_face.x
                    stored_face_y = stored_face.y
                    stored_face_w = stored_face.w
                    stored_face_h = stored_face.h

                    for (x, y, w, h) in faces_locations_:
                       
                        # Check if the detected face overlaps with any stored face
                        if x < stored_face_x + stored_face_w and \
                           x + w > stored_face_x and \
                           y < stored_face_y + stored_face_h and \
                           y + h > stored_face_y:
                            # If overlapping, add criminal details to the list
                            photo_base64 = base64.b64encode(stored_face.face_image).decode('utf-8')
                            criminal.append({
                                'photo': photo_base64,
                                'name': stored_face.name,
                                'date_of_birth': stored_face.date_of_birth,
                                'gender': stored_face.gender,
                                'national_verification_number': stored_face.national_verification_number,
                                'crime_type': stored_face.crime_type,
                                'age': stored_face.age
                            })

                if not criminal:
                    flash('No Criminals Detected.', 'error')

                return render_template('img_inv.html', criminals=criminal, img_base64=img_base64)
    return redirect(url_for('img_inv'))               



@app.route('/inv_vd')
def img_vd():
    return render_template('inv_vd.html')


@app.route('/inv_vd', methods=['POST'])
def investigate_video():
    criminal = []

    if request.method == 'POST':
        video_file = request.files.get('video')
        if not video_file:
            flash('No Video Uploaded.', 'error')
            return render_template('inv_vd.html')
        elif video_file:
            # Save the uploaded video file to a temporary file
            temp_video_file = tempfile.NamedTemporaryFile(delete=False)
            video_file.save(temp_video_file.name)
            video_path = temp_video_file.name

            custom_video_capture = CustomVideoCapture()
            custom_video_capture.open(video_path)

            # Frame skipping parameters
            skip_frames = 15  # Skip every 5th frame
            frame_count = 0

            # number of frames with detected face
            fwfd = 0
            # Flag to indicate if any criminals are detected
            criminals_detected = False

            # Load the Haar cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Retrieve stored faces from the database
            stored_faces = DetectedCriminals.query.all()
            if len(stored_faces) == 0:
                flash('No Criminal Faces Stored In The Database.', 'error')
            else:
                # Frame processing loop
                while True:
                    ret, frame = custom_video_capture.read()
                    if not ret:
                        break

                    if frame_count % skip_frames != 0:
                        frame_count += 1
                        continue  # Skip this frame

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Detect faces in the frame using Haar cascade
                    faces_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    # If no faces are detected in the frame, continue to the next frame
                    if len(faces_locations) == 0:
                        frame_count += 1
                        continue
                    else:
                        fwfd += 1
                        for (x, y, w, h) in faces_locations:
                            # Check if the detected face overlaps with any stored face
                            for stored_face in stored_faces:
                                stored_face_x = stored_face.x
                                stored_face_y = stored_face.y
                                stored_face_w = stored_face.w
                                stored_face_h = stored_face.h
                                if x < stored_face_x + stored_face_w and \
                                   x + w > stored_face_x and \
                                   y < stored_face_y + stored_face_h and \
                                   y + h > stored_face_y:
                                    # If overlapping, add criminal details to the list
                                    photo_base64 = base64.b64encode(stored_face.face_image).decode('utf-8')
                                    criminal.append({
                                        'photo': photo_base64,
                                        'name': stored_face.name,
                                        'date_of_birth': stored_face.date_of_birth,
                                        'gender': stored_face.gender,
                                        'national_verification_number': stored_face.national_verification_number,
                                        'crime_type': stored_face.crime_type,
                                        'age': stored_face.age
                                    })
                                    criminals_detected = True

                    frame_count += 1

                custom_video_capture.release()

                if fwfd == 0:
                    flash('No Faces Detected In The Uploaded Video.', 'error')
                else:
                    if not criminals_detected:
                        flash('No Criminals Detected.', 'error')

    return render_template('inv_vd.html', criminals=criminal)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
