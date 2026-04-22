import os
import sys
import json
import logging
import cv2
import random

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances

setup_logger()
logger = logging.getLogger("sagemaker")
logger.setLevel(logging.INFO)

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomLighting(0.5),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

def get_num_classes(annot_path):
    with open(annot_path, 'r') as f:
        coco = json.load(f)
    return len(coco.get("categories", []))

def main():
    # SageMaker automatically downloads S3 data to the SM_CHANNEL_TRAIN directory
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    logger.info(f"Using dataset from: {data_dir}")
    
    # We expect `train` and `test` directories to exist inside the downloaded data
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    train_annot = os.path.join(train_dir, "annotations.json")
    test_annot = os.path.join(test_dir, "annotations.json")
    
    if not os.path.exists(train_annot):
        logger.error(f"Cannot find training annotations at {train_annot}")
        sys.exit(1)
        
    num_classes = get_num_classes(train_annot)
    logger.info(f"Found {num_classes} categories.")
    
    # Register the datasets using native COCO paths
    register_coco_instances("my_dataset_train", {}, train_annot, train_dir)
    register_coco_instances("my_dataset_test", {}, test_annot, test_dir)
    
    # Load Detectron2 Defaults
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Load initialization weights from Model Zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 30
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Evaluate every 300 iterations
    cfg.TEST.EVAL_PERIOD = 300
    
    # Output to SM_MODEL_DIR so SageMaker natively tars and ships back to S3
    cfg.OUTPUT_DIR = model_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = MyTrainer(cfg)
    
    # Register Checkpointer to save best model based on AP50
    best_checkpointer = hooks.BestCheckpointer(
        eval_period=cfg.TEST.EVAL_PERIOD,
        checkpointer=trainer.checkpointer,
        val_metric="bbox/AP50",
        mode="max",
        file_prefix="model_best"
    )
    trainer.register_hooks([best_checkpointer])
    
    logger.info("Starting Training...")
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    logger.info("Training complete. Models saved successfully in container.")

if __name__ == "__main__":
    main()
