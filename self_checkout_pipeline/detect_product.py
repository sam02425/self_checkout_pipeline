import cv2
import tensorflow as tf
import numpy as np

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def detect_product(model, camera_ids):
    cameras = [cv2.VideoCapture(cam_id) for cam_id in camera_ids]
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

        cv2.imshow('Product Detection', frames[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model('models/trained_model')
    detect_product(model, [0, 1, 2])
