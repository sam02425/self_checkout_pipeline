import os

# Define the directory structure
directories = [
    "self_checkout_pipeline",
    "self_checkout_pipeline/.github/workflows",
    "self_checkout_pipeline/config",
    "self_checkout_pipeline/data/captured_images",
    "self_checkout_pipeline/data/annotated_images",
    "self_checkout_pipeline/data/augmented_images",
    "self_checkout_pipeline/data/tfrecord",
    "self_checkout_pipeline/data/train",
    "self_checkout_pipeline/data/val",
    "self_checkout_pipeline/models/trained_model"
]

# Define the files to be created
files = [
    "self_checkout_pipeline/capture_images.py",
    "self_checkout_pipeline/annotate_images.py",
    "self_checkout_pipeline/prepare_dataset.py",
    "self_checkout_pipeline/convert_to_tfrecord.py",
    "self_checkout_pipeline/train_model.py",
    "self_checkout_pipeline/evaluate_model.py",
    "self_checkout_pipeline/deploy_model.py",
    "self_checkout_pipeline/requirements.txt",
    "self_checkout_pipeline/Dockerfile",
    "self_checkout_pipeline/.github/workflows/ci_cd_pipeline.yml",
    "self_checkout_pipeline/config/pipeline.config"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Create files
for file in files:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            pass  # Just create an empty file

print("Directory structure created successfully.")