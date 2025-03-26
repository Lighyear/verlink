from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from pygments.lexers.capnproto import CapnProtoLexer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import datetime
import sqlite3
import os

app = Flask(__name__)
# Load the trained CNN model
try:
    model = load_model('fer_ck_cnn_improved_model.h5')
except FileNotFoundError:
    print(
        "Error: 'fer_ck_cnn_model.h5' not found in the project directory. Please run 'train_model.py' to generate the model.")
    exit(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Initialize video capture
cap = None
is_running = False
is_paused = False
out = None
recorded_data = []
emotion_history = []
start_time = None
recording_id = None
frame_width = 640
frame_height = 480
display_width = frame_width + 200
display_height = frame_height
fps = 30


# Database functions
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recordings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    duration REAL NOT NULL
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id INTEGER,
                    timestamp TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (recording_id) REFERENCES recordings(id)
                 )''')
    conn.commit()
    conn.close()


# Create the recordings directory if it doesn't exist
if not os.path.exists('static/recordings'):
    os.makedirs('static/recordings')

init_db()


def initialize_video():
    global cap, out, recorded_data, emotion_history, start_time, is_running, is_paused, recording_id
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Trying alternative indices...")
        for i in range(1, 3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Success: Webcam opened with index {i}.")
                break
        if not cap.isOpened():
            print(
                "Error: Could not open any webcam. Please ensure a webcam is connected and not in use by another application.")
            return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Generate a unique filename for the video
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    video_filename = f'recorded_emotion_video_{timestamp}.avi'
    video_path = os.path.join('static/recordings', video_filename)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (display_width, display_height))

    recorded_data = []
    emotion_history = []
    start_time = time.time()
    is_running = True
    is_paused = False

    # Insert recording metadata into the database
    conn = get_db_connection()
    c = conn.cursor()
    start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO recordings (file_path, start_time, duration) VALUES (?, ?, 0)",
              (video_path, start_time_str))
    recording_id = c.lastrowid
    conn.commit()
    conn.close()

    print("Video initialized successfully.")
    return True


def generate_frames():
    global recorded_data, emotion_history, is_running, is_paused, recording_id
    print("Starting frame generation...")
    while is_running:
        if is_paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        display_frame[0:frame_height, 0:frame_width] = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            prediction = model.predict(face_roi)[0]
            emotion_label = np.argmax(prediction)
            emotion = emotion_map[emotion_label]
            confidence = prediction[emotion_label]

            print(f"Detected emotion: {emotion} with confidence {confidence:.2f}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            recorded_data.append({
                'recording_id': recording_id,
                'timestamp': timestamp,
                'emotion': emotion,
                'confidence': confidence
            })

            emotion_history.append(f"{emotion} ({confidence:.2f})")
            if len(emotion_history) > 10:
                emotion_history.pop(0)

        display_frame[0:frame_height, 0:frame_width] = frame
        for i, emotion_text in enumerate(emotion_history):
            y_pos = 30 + i * 30
            cv2.putText(display_frame, emotion_text, (frame_width + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(display_frame)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            print("Error: Failed to encode frame to JPEG.")
            continue
        frame = buffer.tobytes()
        print("Yielding frame...")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    print("Accessing the description page...")
    return render_template('index.html')


@app.route('/video')
def video():
    return render_template('video.html')


@app.route('/recordings')
def recordings():
    conn = get_db_connection()
    recordings = conn.execute('SELECT * FROM recordings ORDER BY start_time DESC').fetchall()
    conn.close()
    return render_template('recordings.html', recordings=recordings)


@app.route('/video_feed')
def video_feed():
    if not is_running:
        print("Error: Video feed requested but video is not running.")
        return Response("Video not started.", status=400)
    print("Video feed route accessed. Starting to stream frames...")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start():
    global is_running
    if not is_running:
        if initialize_video():
            return "Video started."
        return "Failed to start video.", 500
    return "Video already running."


@app.route('/pause', methods=['POST'])
def pause():
    global is_paused
    if is_running:
        is_paused = not is_paused
        return "Paused" if is_paused else "Resumed"
    return "Video not running.", 400


@app.route('/close', methods=['POST'])
def close():
    global is_running, is_paused, cap, out, recording_id
    if is_running:
        is_running = False
        is_paused = False
        duration = time.time() - start_time
        cap.release()
        out.release()

        # Update recording duration in the database
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("UPDATE recordings SET duration = ? WHERE id = ?", (duration, recording_id))

        # Insert emotion data into the database
        for data in recorded_data:
            c.execute("INSERT INTO emotions (recording_id, timestamp, emotion, confidence) VALUES (?, ?, ?, ?)",
                      (data['recording_id'], data['timestamp'], data['emotion'], data['confidence']))
        conn.commit()
        conn.close()

        return "Recording stopped."
    return "Video not running.", 400


if __name__ == '__main__':
    app.run(debug=True)