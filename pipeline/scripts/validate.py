import os
import sys
import json
import tarfile
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# The pipeline mounts the S3 input data to this local container path
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

def validate_data():
    logger.info("Starting Data Validation...")
    # Expected structure: a tar.gz file or a directory with images/ and annotations.json
    
    # We'll assume the input is uncompressed by SageMaker, or we extract it if it's a tar
    extracted_dir = INPUT_DIR
    
    # Let's find annotations.json
    annot_path = None
    for root, dirs, files in os.walk(INPUT_DIR):
        if "annotations.json" in files:
            annot_path = os.path.join(root, "annotations.json")
            extracted_dir = root
            break
        elif any(f.endswith('.tar.gz') for f in files):
            # If it's a tar.gz, extract it first
            tar_name = [f for f in files if f.endswith('.tar.gz')][0]
            tar_path = os.path.join(root, tar_name)
            logger.info(f"Extracting {tar_path}")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=OUTPUT_DIR)
            extracted_dir = OUTPUT_DIR
            annot_path = os.path.join(OUTPUT_DIR, "annotations.json")
            break

    if not annot_path or not os.path.exists(annot_path):
        logger.error("annotations.json not found!")
        sys.exit(1)

    try:
        with open(annot_path, 'r') as f:
            coco = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse annotations.json: {e}")
        sys.exit(1)
        
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    
    if not images or not annotations or not categories:
        logger.error("COCO JSON is missing images, annotations, or categories.")
        sys.exit(1)
        
    logger.info(f"Found {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    
    # Check if images exist
    missing_images = 0
    for img in images:
        img_path = os.path.join(extracted_dir, img["file_name"])
        if not os.path.exists(img_path):
            missing_images += 1
            
    if missing_images > 0:
        logger.error(f"Validation Failed: {missing_images} missing images out of {len(images)}.")
        sys.exit(1)
        
    logger.info("Data Validation Passed.")
    
    # We save a success flag to move to the next steps
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "validation_status.txt"), "w") as f:
        f.write("SUCCESS\n")
        f.write(f"dataset_dir={extracted_dir}\n")

if __name__ == "__main__":
    validate_data()
