from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Optional, Union
from slimdoc import DATASET_CONF, DEFAULTS, ENV, SUPPORTED_DATASET, DUModel, DistillMode


@dataclass(frozen=True)  # frozen=True allows us to use the type as dict key
class ExperimentDefinition:
    # Model config
    model: DUModel
    mode: DistillMode

    # Data
    dataset: SUPPORTED_DATASET
    use_chatgpt_labels: bool

    # Distillation
    student_layer_map: tuple[int]  # default: [2,5,8,11]
    vocab_name: str  # Filename of the vocab to use (without file ending), if None use the original vocab

    # Parameters which are filled during execution
    run_name_suffix: str = None
    alpha: float = DEFAULTS.ALPHA  # Distillation hidden state loss weight
    beta: float = DEFAULTS.BETA  # Distillation embedding loss weight
    gamma: float = DEFAULTS.GAMMA  # Distillation attention loss weight
    delta: float = DEFAULTS.DELTA  # Distillation logits loss weight

    use_ce_loss_for_logits_distillation: bool = False

    @property
    def num_epochs(self) -> int:
        return get_num_epochs(dataset=self.dataset, mode=self.mode)

    @property
    def run_name(self) -> str:
        rn = get_run_name(
            model=self.model,
            mode=self.mode,
            teacher_run_name=self.teacher_run_name,
            dataset=self.dataset,
            use_chatgpt_labels=self.use_chatgpt_labels,
            student_layer_map=self.student_layer_map,
            vocab_name=self.vocab_name,
            num_epochs=self.num_epochs,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            use_ce_loss_for_logits_distillation=self.use_ce_loss_for_logits_distillation,
        )
        if self.run_name_suffix is not None and self.run_name_suffix != "":
            return f"{rn}_{self.run_name_suffix}"
        else:
            return rn

    @property
    def teacher_run_name(self) -> Optional[str]:
        if (
            self.mode == DistillMode.FINETUNE_TEACHER
            or self.mode == DistillMode.FINETUNE_STUDENT_NEWINIT
        ):
            return None
        elif (
            self.mode == DistillMode.FINETUNE_STUDENT
            or self.mode == DistillMode.DISTILLATION
            or self.mode == DistillMode.DISTILLATION_IN_1PHASE
        ):
            trn = get_teacher_run_name(
                model=self.model,
                dataset=self.dataset,
                use_chatgpt_labels=self.use_chatgpt_labels,
            )
            # if self.run_name_suffix is not None and self.run_name_suffix != "":
            #     return f"{trn}_{self.run_name_suffix}"
            # else:
            #     return trn
            return trn

    @property
    def checkpoint_path(self) -> Path:
        return ENV.MODELS_DIR / self.run_name

    @property
    def task(self) -> str:
        return DATASET_CONF[self.dataset].task

    @property
    def num_labels(self) -> str:
        return DATASET_CONF[self.dataset].num_labels

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(_dict: dict):
        def dataclass_from_dict(klass, d):
            try:
                fieldtypes = {f.name: f.type for f in dataclasses.fields(klass)}
                return klass(**{f: dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
            except:
                return d  # Not a dataclass field

        return dataclass_from_dict(ExperimentDefinition, _dict)


def get_num_epochs(dataset: SUPPORTED_DATASET, mode: DistillMode):
    _30_epoch_ds: set[SUPPORTED_DATASET] = {
        "DocVQA",
        "InfographicsVQA",
        "WikiTableQuestions",
    }
    _50_epoch_ds: set[SUPPORTED_DATASET] = {
        "SROIE",
        "FUNSD",
        "Insiders P2P",
        "Insiders Rechnungen",
        "Insiders Formulare",
    }

    num_epochs = -1
    if dataset in _30_epoch_ds:
        num_epochs = 30
    elif dataset in _50_epoch_ds:
        num_epochs = 50
    else:
        raise ValueError(f'Unknown number of epochs for dataset "{dataset}"')

    # Distillation has double the number of epochs because training is split in two phases.
    # For normal distillation, this is already handled by the train loop.
    # For the special 1-phase distillation we manually need to increase epoch size
    if mode == DistillMode.DISTILLATION_IN_1PHASE:
        num_epochs *= 2

    return num_epochs


def get_run_name(
    model: DUModel,
    mode: DistillMode,
    teacher_run_name: Optional[str],
    dataset: SUPPORTED_DATASET,
    use_chatgpt_labels: bool,
    student_layer_map: tuple[int],
    vocab_name: Optional[str],
    num_epochs: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    use_ce_loss_for_logits_distillation: bool = False,
) -> str:
    # Model
    modelname = model.name.replace("_", "-")

    # Mode
    modename = None
    if mode == DistillMode.FINETUNE_TEACHER:
        modename = "ft-teacher"
    elif mode == DistillMode.FINETUNE_STUDENT:
        modename = "ft-student"
    elif mode == DistillMode.FINETUNE_STUDENT_NEWINIT:
        modename = "ft-student-newinit"
    elif mode == DistillMode.DISTILLATION:
        modename = "dt-student"
    elif mode == DistillMode.DISTILLATION_IN_1PHASE:
        modename = "dt-student-1phase"
    else:
        raise ValueError(f"Unknown DistillMode: {mode}")

    # Dataset
    dsname = f'{dataset.lower()}{"-chatgpt" if use_chatgpt_labels else ""}'

    # Student Layer Mapping
    layersname = None
    if mode != DistillMode.FINETUNE_TEACHER:
        layersname = f"layers-{'-'.join([str(l) for l in student_layer_map])}"

    # Vocab
    vocabname = None
    if vocab_name is None:
        vocabname = None
    elif vocab_name == DEFAULTS.SMALL_VOCAB:
        vocabname = "smallvocab"
    elif vocab_name == DEFAULTS.TINY_VOCAB:
        vocabname = "tinyvocab"
    else:
        raise ValueError(f'Unknown vocab "{vocab_name}"')

    # Num Epochs
    if mode == DistillMode.DISTILLATION:
        epochsname = f"{2*num_epochs}epochs"
    else:
        # For DistillMode.DISTILLATION_IN_1PHASE the number of epochs is already doubled in get_num_epochs
        epochsname = f"{num_epochs}epochs"

    # alpha beta gamma delta
    loss_weights_str = None
    if mode == DistillMode.DISTILLATION or mode == DistillMode.DISTILLATION_IN_1PHASE:
        loss_weights_str = f"alpha={alpha}-beta={beta}-gamma={gamma}-delta={delta}"

    ce_loss_str = None
    if use_ce_loss_for_logits_distillation:
        ce_loss_str = "logits-distill-ce-loss"

    components = [
        modelname,
        modename,
        dsname,
        layersname,
        vocabname,
        epochsname,
        loss_weights_str,
        ce_loss_str,
    ]
    run_name = "_".join([x for x in components if x is not None])

    return run_name


def get_teacher_run_name(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    return get_run_name(
        model=model,
        mode=DistillMode.FINETUNE_TEACHER,
        teacher_run_name=None,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=None,
        vocab_name=None,
        num_epochs=get_num_epochs(dataset=dataset, mode=DistillMode.FINETUNE_TEACHER),
        alpha=None,
        beta=None,
        gamma=None,
        delta=None,
    )
