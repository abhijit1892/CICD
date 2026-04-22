import os
import sys
import json
import logging
import cv2
import tarfile

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

setup_logger()
logger = logging.getLogger("sagemaker")
logger.setLevel(logging.INFO)

# SageMaker Processing Paths
INPUT_DIR = "/opt/ml/processing/input"
MODEL_DIR = "/opt/ml/processing/model"
EVAL_DIR = "/opt/ml/processing/evaluation"

def evaluate():
    logger.info("Starting Evaluation...")
    
    # Extract data if needed, or find annotations.json
    annot_path = None
    data_dir = INPUT_DIR
    for root, dirs, files in os.walk(INPUT_DIR):
        if "annotations.json" in files:
            annot_path = os.path.join(root, "annotations.json")
            data_dir = root
            break
        elif any(f.endswith('.tar.gz') for f in files):
            tar_name = [f for f in files if f.endswith('.tar.gz')][0]
            with tarfile.open(os.path.join(root, tar_name), "r:gz") as tar:
                tar.extractall(path="/tmp/eval_data")
            data_dir = "/tmp/eval_data"
            annot_path = os.path.join(data_dir, "annotations.json")
            break

    if not annot_path:
        logger.error("annotations.json not found!")
        sys.exit(1)

    # Register the dataset (treating the entire upload as test set for evaluation)
    register_coco_instances("eval_dataset", {}, annot_path, data_dir)
    
    # Load model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # Assume 6 classes or get from annotations
    with open(annot_path) as f:
        coco = json.load(f)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(coco.get("categories", []))
    
    # The Model from TrainingStep is downloaded to MODEL_DIR
    # It might be model_best.pth or model_final.pth
    model_weights = os.path.join(MODEL_DIR, "model_best.pth")
    if not os.path.exists(model_weights):
        model_weights = os.path.join(MODEL_DIR, "model_final.pth")
    
    logger.info(f"Using model weights: {model_weights}")
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator("eval_dataset", output_dir="/tmp/output")
    val_loader = build_detection_test_loader(cfg, "eval_dataset")
    
    logger.info("Running inference on dataset...")
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    logger.info(f"Metrics: {metrics}")
    
    map_50 = metrics.get("bbox", {}).get("AP50", 0)
    
    # Save to evaluation.json for ConditionStep
    report_dict = {
        "metrics": {
            "mAP50": {
                "value": float(map_50),
                "standard_deviation": "NaN"
            }
        }
    }
    
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "evaluation.json"), "w") as f:
        json.dump(report_dict, f)
        
    logger.info(f"Saved evaluation report with mAP50 = {map_50}")

if __name__ == "__main__":
    evaluate()
