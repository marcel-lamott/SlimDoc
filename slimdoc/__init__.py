import dataclasses
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Union
from transformers import LiltModel, LiltConfig, LayoutLMv3Model, LayoutLMv3Config


_root_path = Path(__file__).parent.parent.resolve()


# Project paths
class ENV:
    ROOT_DIR: Path = _root_path
    DATA_DIR: Path = ROOT_DIR / "data"

    # Contains the raw datasets in their original format
    DATASETS_DIR: Path = DATA_DIR / "datasets"

    # Contains the preprocesses datasets (tokenized, all splits, bboxes normalized)
    PROCESSED_DATASETS_DIR: Path = DATASETS_DIR / "processed"

    # Contains the created vocabularies
    VOCABS_DIR: Path = DATA_DIR / "vocab"

    MODELS_DIR: Path = DATA_DIR / "models"

    # Contains temporary exports which can be deleted anytime, e.g. for DUE evaluation
    TEMP_DIR: Path = DATA_DIR / "temp"

    EVAL_RESULTS_DIR: Path = DATA_DIR / "results"

    EVAL_ANALYZATION_DIR: Path = EVAL_RESULTS_DIR / "analyzed"

    RUNNER_RESULTS_DIR: Path = DATA_DIR / "runner" / "results"


# Create the project structure
ENV.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ENV.PROCESSED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ENV.VOCABS_DIR.mkdir(parents=True, exist_ok=True)
ENV.MODELS_DIR.mkdir(parents=True, exist_ok=True)
ENV.TEMP_DIR.mkdir(parents=True, exist_ok=True)
ENV.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ENV.EVAL_ANALYZATION_DIR.mkdir(parents=True, exist_ok=True)
ENV.RUNNER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


SUPPORTED_MODEL_IDS = Literal[
    "SCUT-DLVCLab/lilt-roberta-en-base", "microsoft/layoutlmv3-base"
]
SUPPORTED_CONFIG = Union[LiltConfig, LayoutLMv3Config]
SUPPORTED_DATASET = Literal[
    "DocVQA", "InfographicsVQA", "SROIE", "WikiTableQuestions", "FUNSD"
]


# Static read-only configurations
class CONSTS:
    WANDB_PROJECT_ID: str = "SlimDoc"
    LILT_HF_MODEL_ID: str = "SCUT-DLVCLab/lilt-roberta-en-base"
    LAYOUTLMV3_HF_MODEL_ID: str = "microsoft/layoutlmv3-base"


# Default values for Training
class DEFAULTS:
    # Student model
    DEFAULT_LAYER_MAP = tuple([0, 3, 6, 9])

    # Vocab
    SMALL_VOCAB = "DocVQA_InfographicsVQA-train-15000"
    TINY_VOCAB = "DocVQA_InfographicsVQA-train-5000"

    # Dataloading
    BATCH_SIZE = 16
    DATALOADER_NUM_WORKERS = 32

    # LR
    LR_START: float = 1e-5
    LR_END: float = 1e-8
    LR_SCHEDULE_WARMUP_STEPS_FRAC_OF_TOTAL: float = 0.1

    # Loss weights
    ALPHA = 1
    BETA = 1
    GAMMA = 100
    DELTA = 0.01


# Task identifiers
class TASKS:
    VQA = "VQA"
    SER = "SER"


class DistillMode(Enum):
    FINETUNE_TEACHER = 1
    FINETUNE_STUDENT = 2
    DISTILLATION = 3
    # Ablations
    DISTILLATION_IN_1PHASE = 4
    FINETUNE_STUDENT_NEWINIT = 5


class DUModel(Enum):
    LiLT_TextFlow = (
        1  # default LiLT model, hidden states are only returned for text flow
    )
    LiLT_TextAndLayoutFlow = 2  # customized LiLT model, which outputs hidden states also for layout flow (as well as attention scores but there are same for both flows)
    LayoutLMv3_TextAndImage = 3  # default LayoutLMv3 model
    LayoutLMv3_TextOnly = (
        4  # default LayoutLMv3 model, but we pass no pixel_values to the model
    )
    LiLT_TextFlow_NoLayout = (
        5  # baseline for LiLT where we don't input any bounding boxes
    )
    LayoutLMv3_NoVisionNoLayout = 6


@dataclass
class DatasetConfig:
    task: str
    num_labels: int
    eval_split_name: Union[str, None]
    has_test_split: bool


# Dataset specific configurations
DATASET_CONF = {
    "DocVQA": DatasetConfig(
        task=TASKS.VQA,
        num_labels=3,  # None, start, stop
        eval_split_name="dev",
        has_test_split=True,
    ),
    "InfographicsVQA": DatasetConfig(
        task=TASKS.VQA,
        num_labels=3,  # None, start, stop
        eval_split_name="dev",
        has_test_split=True,
    ),
    "WikiTableQuestions": DatasetConfig(
        task=TASKS.VQA,
        num_labels=3,  # None, start, stop
        eval_split_name="dev",
        has_test_split=True,
    ),
    "SROIE": DatasetConfig(
        task=TASKS.SER,
        num_labels=5,  # including None
        eval_split_name=None,
        has_test_split=True,
    ),
    "FUNSD": DatasetConfig(
        task=TASKS.SER,
        num_labels=4,  # {'answer': 0, 'header': 1, 'other': 2, 'question': 3}
        eval_split_name=None,
        has_test_split=True,
    ),
}
