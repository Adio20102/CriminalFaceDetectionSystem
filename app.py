import cv2
import numpy as np
import base64
import face_recognition
import tempfile


from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///criminal_database.db'
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


class DetectedFace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    crime_type = db.Column(db.String(100))
    age = db.Column(db.Integer)
    national_verification_number = db.Column(db.String(100))
    top = db.Column(db.Integer)
    right = db.Column(db.Integer)
    left = db.Column(db.Integer)
    bottom = db.Column(db.Integer)
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

            # Face detection
            img_array = np.frombuffer(photo_data, dtype=np.uint8)  # Convert binary data to numpy array
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode the image
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces in the uploaded image
            face_location = face_recognition.face_locations(rgb_image)
            if not face_location:  # No faces detected
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
                existing_face = DetectedFace.query.filter_by(national_verification_number=national_verification_number).first()

                if existing_face:

                    specific_face_encoding = [np.frombuffer(existing_face.face_image, dtype=np.uint8)]
                    specific_face_encoding = [face_recognition.face_encodings(cv2.imdecode(encoding, cv2.IMREAD_COLOR))[0] for encoding in specific_face_encoding]
                    bool = False
                    # bool is a flag for face match
                    # face encodings of the uploaded image is as follows:
                    face_encods = face_recognition.face_encodings(rgb_image, face_location)
                    for face_encod, (top, right, bottom, left) in zip(face_encods, face_location):
                        face_matching = face_recognition.compare_faces(specific_face_encoding, face_encod)
                        if any(face_matching):
                            bool = True

                    if bool == True:
                        if existing_face.date_of_birth == date_of_birth.date() and existing_face.name == name and existing_face.gender == gender:

                           if int(age) >= int(existing_face.age):
                              existing_face.crime_type = existing_face.crime_type + ',' + crime_type
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
                    else:
                        flash(f'A Criminal Already With Different Face Exists With National Verification Number: {existing_face.national_verification_number}' , 'info')
                        return render_template('register_form.html')

                # To check is the face uploaded image is same as that of a registered criminal in the database
                stored_criminal_faces_ = DetectedFace.query.all()
                if stored_criminal_faces_:
                    criminal_faces_encodings = [np.frombuffer(face.face_image, dtype=np.uint8) for face in stored_criminal_faces_]
                    criminal_faces_encodings = [face_recognition.face_encodings(cv2.imdecode(encoding, cv2.IMREAD_COLOR))[0] for encoding in criminal_faces_encodings]

                    face_encodins = face_recognition.face_encodings(rgb_image, face_location)

                    for face_encodin, (top, right, bottom, left) in zip(face_encodins, face_location):
                        match = face_recognition.compare_faces(criminal_faces_encodings, face_encodin)
                        if any(match):
                            flash('Criminal With Same Face Already Exists.', 'warning')
                            return render_template('register_form.html')


                for (top, right, bottom, left) in face_location:
                    cropped_face = img[top:bottom, left:right]
                    # Convert image to byte array before storing
                    image_as_bytes = cv2.imencode('.jpg', cropped_face)[1].tobytes()
                    new_face = DetectedFace(
                        name=name,
                        date_of_birth=date_of_birth,
                        age=age,
                        gender=gender,
                        crime_type=crime_type,
                        national_verification_number=national_verification_number,
                        top=top,
                        right=right,
                        bottom=bottom,
                        left=left,
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

            # Convert the uploaded photo to numpy array
            nparr = np.frombuffer(photo_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert the image to RGB format (required by face_recognition library)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces in the uploaded image
            face_locations_ = face_recognition.face_locations(rgb_img)

            if len(face_locations_) == 0:
                flash('No faces were detected in the uploaded image.', 'error')
            else:
                # Encode the uploaded image to base64 for displaying
                _, img_encoded = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                # Retrieve stored faces from the database
                stored_faces_ = DetectedFace.query.all()
                if len(stored_faces_) == 0:
                    flash('No faces stored in the database.', 'error')
                    return redirect(url_for('img_inv'))

                # Load stored face encodings
                stored_encoding_ = [np.frombuffer(face.face_image, dtype=np.uint8) for face in stored_faces_]
                stored_encoding_ = [face_recognition.face_encodings(cv2.imdecode(encoding, cv2.IMREAD_COLOR))[0] for encoding in stored_encoding_]

                # Convert face locations to encodings
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations_)

                # Iterate through detected faces
                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations_):
                    # Compare the face encoding with stored encodings
                    matches = face_recognition.compare_faces(stored_encoding_, face_encoding)

                    # Check if any matches found
                    if any(matches):
                        matched_face = stored_faces_[matches.index(True)]
                        photo_base64 = base64.b64encode(matched_face.face_image).decode('utf-8')
                        criminal.append({
                            'photo': photo_base64,
                            'name': matched_face.name,
                            'date_of_birth': matched_face.date_of_birth,
                            'gender': matched_face.gender,
                            'national_verification_number': matched_face.national_verification_number,
                            'crime_type': matched_face.crime_type,
                            'age': matched_face.age,

                        })
                    else:
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

            # Retrieve stored faces from the database
            stored_faces = DetectedFace.query.all()
            if len(stored_faces) == 0:
                flash('No Faces Stored In The Database.', 'error')
                return redirect(url_for('inv_vd'))

            # Load stored face encodings and associated information
            stored_encodings = [np.frombuffer(face.face_image, dtype=np.uint8) for face in stored_faces]
            stored_encodings = [face_recognition.face_encodings(cv2.imdecode(encoding, cv2.IMREAD_COLOR))[0] for
                                encoding in stored_encodings]

            # Frame skipping parameters
            skip_frames = 15  # Skip every 5th frame
            frame_count = 0

            # number of frames with detected face
            fwfd = 0
            # Flag to indicate if any criminals are detected
            criminals_detected = False
            # Face detection and recognition logic here
            while True:
                ret, frame = custom_video_capture.read()
                if not ret:
                    break

                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue  # Skip this frame

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                # If no faces are detected in the frame, continue to the next frame
                if not face_locations:
                    frame_count += 1
                    continue
                else:
                    fwfd += 1
                    CriminalDetected = False
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                        # Compare the detected face encoding with the stored encodings
                        matches = face_recognition.compare_faces(stored_encodings, encoding)
                        if any(matches):
                            matched_index = matches.index(True)
                            # Gather information about the detected criminal
                            criminal_info = {
                                'name': stored_faces[matched_index].name,
                                'date_of_birth': stored_faces[matched_index].date_of_birth,
                                'gender': stored_faces[matched_index].gender,
                                'national_verification_number': stored_faces[
                                    matched_index].national_verification_number,
                                'crime_type': stored_faces[matched_index].crime_type,
                                'age': stored_faces[matched_index].age,
                                'photo': base64.b64encode(stored_faces[matched_index].face_image).decode('utf-8')
                            }
                            criminal.append(criminal_info)
                            criminals_detected = True
                frame_count += 1
            custom_video_capture.release()

            if fwfd == 0:
                flash('No Faces Detected In The Uploaded Video.', 'error')
            else:
                if not criminals_detected:
                    flash('No Criminals Detected.', 'error')
    return render_template('inv_vd.html', criminals=criminal, )



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
