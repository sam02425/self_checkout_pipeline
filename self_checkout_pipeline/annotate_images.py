import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os

def annotate_images(image_dir, output_dir, model_type='vit_h', checkpoint='sam_vit_h_4b8939.pth'):
    os.makedirs(output_dir, exist_ok=True)
    sam = sam_model_registrymodel_type
    predictor = SamPredictor(sam)

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        predictor.set_image(image)
        masks, _, _ = predictor.predict()

        for mask in masks:
            segmented_image = np.zeros_like(image)
            segmented_image[mask] = image[mask]
            cv2.imwrite(os.path.join(output_dir, f'seg_{img_file}'), segmented_image)

if __name__ == "__main__":
    annotate_images('data/captured_images', 'data/annotated_images')
