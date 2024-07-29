import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def prepare_dataset(image_dir, output_dir, augment=False):
    os.makedirs(output_dir, exist_ok=True)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if augment else ImageDataGenerator()

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        image = tf.keras.preprocessing.image.load_img(img_path)
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 20:
                break

if __name__ == "__main__":
    prepare_dataset('data/annotated_images', 'data/augmented_images', augment=True)
