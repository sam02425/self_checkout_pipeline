import cv2
import os

def capture_images(camera_ids, output_dir, num_images=100):
    os.makedirs(output_dir, exist_ok=True)
    cameras = [cv2.VideoCapture(cam_id) for cam_id in camera_ids]

    for i in range(num_images):
        for cam_idx, cam in enumerate(cameras):
            ret, frame = cam.read()
            if ret:
                cv2.imwrite(os.path.join(output_dir, f'cam{cam_idx}_img{i}.jpg'), frame)
            else:
                print(f"Failed to capture image from camera {cam_idx}")

    for cam in cameras:
        cam.release()

if __name__ == "__main__":
    capture_images([0, 1, 2], 'data/captured_images')
