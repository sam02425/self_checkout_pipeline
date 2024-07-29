import tensorflow as tf
import os

def create_tf_example(image_path, label):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string)
    height, width, _ = image.shape

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode()])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string.numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
    return tf_example

def convert_to_tfrecord(image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        label = 0  # Replace with actual label
        tf_example = create_tf_example(img_path, label)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    convert_to_tfrecord('data/augmented_images', 'data/tfrecord/dataset.tfrecord')
