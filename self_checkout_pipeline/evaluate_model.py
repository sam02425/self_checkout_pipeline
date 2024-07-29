import tensorflow as tf

def evaluate_model(model_dir, validation_data):
    model = tf.keras.models.load_model(model_dir)
    results = model.evaluate(validation_data)
    print(f"Validation Results: {results}")

if __name__ == "__main__":
    evaluate_model('models/trained_model', 'data/val')
