import os
import sys
import json
import logging
import cv2
import random
import shutil

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

def prepare_data(data_dir, work_dir):
    logger.info(f"Preparing data from {data_dir} to {work_dir}")
    train_dir = os.path.join(work_dir, "train")
    test_dir = os.path.join(work_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    annot_path = None
    for root, dirs, files in os.walk(data_dir):
        if "annotations.json" in files:
            annot_path = os.path.join(root, "annotations.json")
            data_dir = root
            break
            
    with open(annot_path) as f:
        coco = json.load(f)
        
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    random.shuffle(images)
    split_ratio = 0.8
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    train_ids = set([img["id"] for img in train_images])
    test_ids = set([img["id"] for img in test_images])

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
    test_annotations = [ann for ann in annotations if ann["image_id"] in test_ids]

    for img in train_images:
        shutil.copy(os.path.join(data_dir, img["file_name"]), os.path.join(train_dir, img["file_name"]))
    for img in test_images:
        shutil.copy(os.path.join(data_dir, img["file_name"]), os.path.join(test_dir, img["file_name"]))

    train_coco = {"images": train_images, "annotations": train_annotations, "categories": categories}
    test_coco = {"images": test_images, "annotations": test_annotations, "categories": categories}

    with open(os.path.join(train_dir, "_annotations.coco.json"), "w") as f:
        json.dump(train_coco, f)
    with open(os.path.join(test_dir, "_annotations.coco.json"), "w") as f:
        json.dump(test_coco, f)
        
    return train_dir, test_dir, len(categories)

def main():
    # SageMaker paths
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    output_data_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    # We use a local temp dir for data prep
    work_dir = "/tmp/data"
    train_dir, test_dir, num_classes = prepare_data(data_dir, work_dir)
    
    register_coco_instances("my_dataset_train", {}, os.path.join(train_dir, "_annotations.coco.json"), train_dir)
    register_coco_instances("my_dataset_test", {}, os.path.join(test_dir, "_annotations.coco.json"), test_dir)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    cfg.TEST.EVAL_PERIOD = 300
    
    # Set output dir to SM_MODEL_DIR so trained weights are packaged into model.tar.gz by SageMaker
    cfg.OUTPUT_DIR = model_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = MyTrainer(cfg)
    
    best_checkpointer = hooks.BestCheckpointer(
        eval_period=cfg.TEST.EVAL_PERIOD,
        checkpointer=trainer.checkpointer,
        val_metric="bbox/AP50",
        mode="max",
        file_prefix="model_best"
    )
    trainer.register_hooks([best_checkpointer])
    
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    logger.info("Training complete. Artifacts saved in SM_MODEL_DIR.")
    
if __name__ == "__main__":
    main()
