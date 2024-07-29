import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os

def train_model(config_path, model_dir, num_steps=1000):
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    model_fn = model_builder.build(model_config=model_config, is_training=True)

    # Load dataset and prepare for training
    raw_dataset = tf.data.TFRecordDataset('data/tfrecord/dataset.tfrecord')
    def _parse_function(proto):
        keys_to_features = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        image = tf.image.decode_jpeg(parsed_features['image/encoded'])
        label = parsed_features['image/class/label']
        return image, label

    dataset = raw_dataset.map(_parse_function)
    train_data = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # Define hyperparameter search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([8, 16, 32]),
        "optimizer": tune.choice(["adam", "sgd"])
    }

    def train_fn(config):
        # Set up model with given hyperparameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]) if config["optimizer"] == "adam" else tf.keras.optimizers.SGD(learning_rate=config["learning_rate"])
        model_fn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

        # Train the model
        model_fn.fit(
            train_data,
            epochs=num_steps,
            batch_size=config["batch_size"],
            validation_data=val_data,
            callbacks=[tensorboard_callback]
        )

        # Save the model
        model_fn.save(model_dir)

    # Hyperparameter tuning with Ray Tune
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=num_steps,
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run(
        train_fn,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=search_space,
        num_samples=10,
        scheduler=scheduler
    )

    # Get the best hyperparameters
    best_config = analysis.get_best_config(metric="accuracy", mode="max")
    print("Best hyperparameters found were: ", best_config)

if __name__ == "__main__":
    train_model('config/pipeline.config', 'models/trained_model')
