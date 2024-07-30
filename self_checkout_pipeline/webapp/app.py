from flask import Flask, render_template, request, redirect, url_for, session, Response
import cv2
import tensorflow as tf
import os
from capture_images import capture_images

app = Flask(__name__)
app.secret_key = 'supersecretkey'

model = tf.keras.models.load_model('../models/trained_model')

def generate_frames():
    cameras = [cv2.VideoCapture(cam_id) for cam_id in [0, 1, 2]]
    detected_products = set()

    while True:
        frames = []
        for cam in cameras:
            ret, frame = cam.read()
            if ret:
                frames.append(frame)

        for frame in frames:
            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = input_tensor[tf.newaxis, ...]

            detections = model(input_tensor)

            for detection in detections['detection_boxes']:
                ymin, xmin, ymax, xmax = detection
                (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                              ymin * frame.shape[0], ymax * frame.shape[0])
                product_id = detections['detection_classes'][0]
                if product_id not in detected_products:
                    detected_products.add(product_id)
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Product {product_id}', (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        ret, buffer = cv2.imencode('.jpg', frames[0])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Simple authentication logic
        if username == 'admin' and password == 'password':
            session['user'] = username
            return redirect(url_for('add_product'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

@app.route('/add_product', methods=['GET', 'POST'])
def add_product():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        product_name = request.form['product_name']
        # Logic to add product to the database
        capture_images([0, 1, 2], 'data/captured_images')
        return redirect(url_for('capture_images'))
    return render_template('add_product.html')

@app.route('/capture_images')
def capture_images_route():
    if 'user' not in session:
        return redirect(url_for('login'))
    # Logic to capture images and start training cycle
    return 'Capturing images and starting training cycle...'

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
