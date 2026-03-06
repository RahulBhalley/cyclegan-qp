# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Config:
    # Data
    DATASET_DIR: str = "datasets"
    DATASET_NAME: str = "vangogh2photo"
    STYLE_NAMES: List[str] = ("ce", "mo", "uk", "vg")
    LOAD_DIM: int = 286
    CROP_DIM: int = 256
    CKPT_DIR: str = "checkpoints"
    SAMPLE_DIR: str = "samples"

    # Quadratic Potential
    LAMBDA: float = 10.0
    NORM: str = "l1"

    # CycleGAN++
    CYC_WEIGHT: float = 10.0
    ID_WEIGHT: float = 0.5

    # Network
    N_CHANNELS: int = 3
    UPSAMPLE: bool = True
    USE_INSTANCE_NORM: bool = True  # Added option for InstanceNorm

    # Training
    RANDOM_SEED: int = 12345
    BATCH_SIZE: int = 4
    LR: float = 2e-4
    BETA1: float = 0.5
    BETA2: float = 0.999
    BEGIN_ITER: int = 0
    END_ITER: int = 15000

    # Optimization
    USE_AMP: bool = True
    USE_COMPILE: bool = True
    MATMUL_PRECISION: str = "high" # 'highest', 'high', 'medium'

    # Inference
    INFER_ITER: int = 15000
    INFER_STYLE: str = "vg"
    IMG_NAME: str = "sun_flower.jpg"
    IN_IMG_DIR: str = "images"
    OUT_STY_DIR: str = "sty"
    OUT_REC_DIR: str = "rec"
    IMG_SIZE: int = None

    # Logs
    ITERS_PER_LOG: int = 100
    ITERS_PER_CKPT: int = 1000

    @property
    def DATASET_PATH(self) -> Dict[str, str]:
        return {
            "trainA": f"./{self.DATASET_DIR}/{self.DATASET_NAME}/trainA",
            "trainB": f"./{self.DATASET_DIR}/{self.DATASET_NAME}/trainB",
            "testA": f"./{self.DATASET_DIR}/{self.DATASET_NAME}/testA",
            "testB": f"./{self.DATASET_DIR}/{self.DATASET_NAME}/testB"
        }

    @property
    def TRAIN_STYLE(self) -> str:
        mapping = {
            "cezanne2photo": "ce",
            "monet2photo": "mo",
            "ukiyoe2photo": "uk",
            "vangogh2photo": "vg"
        }
        return mapping.get(self.DATASET_NAME, "vg")

config = Config()
