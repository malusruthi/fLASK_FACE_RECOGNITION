from flask import Flask, render_template, request
import cv2
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime

app = Flask(__name__)

# Helper functions
def totalreg():
    return len(os.listdir('static/faces'))

def datetoday2():
    return datetime.now().strftime("%d-%B-%Y")

def extract_attendance():
    names = []
    rolls = []
    times = []
    l = 0
    if os.path.exists('Attendance.csv'):
        with open('Attendance.csv', 'r') as f:
            for line in f.readlines():
                entry = line.strip().split(',')
                names.append(entry[0])
                rolls.append(entry[1])
                times.append(entry[2])
                l += 1
    return names, rolls, times, l

def add_attendance(name):
    with open('Attendance.csv', 'a') as f:
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")
        date_string = now.strftime("%d-%B-%Y")
        f.write(f'{name},{date_string},{time_string}\n')

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2())

# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2(), mess='There is no trained model in the static folder. Please add a new face to continue.')

    known_face_encodings = []
    known_face_names = []

    userlist = os.listdir('static/faces')
    for user in userlist:
        user_name = user
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(user_name)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                add_attendance(name)
            
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2())

if __name__ == '__main__':
    app.run(debug=True)
