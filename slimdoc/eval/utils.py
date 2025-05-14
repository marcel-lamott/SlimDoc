import json
from typing import get_args
from slimdoc import ENV, SUPPORTED_DATASET, DUModel, DistillMode, DEFAULTS
from slimdoc.train import ExperimentDefinition
from slimdoc.train.runner import EXPERIMENT_SERIES, get_experiment_definitions


def get_all_required_experiments():
    # All ExperimentDefinitions we have agreed upon to include in the paper
    # (DUModel, collection_name, include_chatgpt_labels)
    return [
        # LiLT-Text
        (DUModel.LiLT_TextFlow, "minimum", True),
        (DUModel.LiLT_TextFlow, "vocab", False),
        (DUModel.LiLT_TextFlow, "ablations", False),
        (DUModel.LiLT_TextFlow, "all_layers", False),
        # LiLT-Text-Layout
        (DUModel.LiLT_TextAndLayoutFlow, "minimum", False),
        (DUModel.LiLT_TextAndLayoutFlow, "vocab", False),
        # LiLT-Text-NoLayout
        (DUModel.LiLT_TextFlow_NoLayout, "minimum", False),
        # LayoutLMv3
        (DUModel.LayoutLMv3_TextAndImage, "minimum", True),
        (DUModel.LayoutLMv3_TextAndImage, "vocab", False),
        (DUModel.LayoutLMv3_TextAndImage, "ablations", False),
        (DUModel.LayoutLMv3_TextAndImage, "all_layers", False),
        # LayoutLMv3-NoVision
        (DUModel.LayoutLMv3_TextOnly, "minimum", False),
    ]


def get_all_datasets(include_chatgpt_labels: bool):
    if include_chatgpt_labels:
        return [
            ("SROIE", False),
            ("FUNSD", False),
            ("DocVQA", False),
            ("InfographicsVQA", False),
            ("WikiTableQuestions", False),
            ("Insiders P2P", False),
            # ("Insiders Rechnungen", False),
            # ("Insiders Formulare", False),
            ("SROIE", True),
            ("DocVQA", True),
            ("InfographicsVQA", True),
            ("WikiTableQuestions", True),
        ]
    else:
        return [
            ("SROIE", False),
            ("FUNSD", False),
            ("DocVQA", False),
            ("InfographicsVQA", False),
            ("WikiTableQuestions", False),
            ("Insiders P2P", False),
            # ("Insiders Rechnungen", False),
            # ("Insiders Formulare", False),
        ]


def get_eds_for_all_datasets(
    collection_name: str, model: DUModel, include_chatgpt_labels: bool
):
    all_ds = get_all_datasets(include_chatgpt_labels=include_chatgpt_labels)
    all_eds = []
    for ds, use_chatgpt_labels in all_ds:
        eds = get_experiment_definitions(
            collection_name=collection_name,
            model=model,
            dataset=ds,
            use_chatgpt_labels=use_chatgpt_labels,
        )
        all_eds.extend(eds)
    return all_eds


# Inverse method to get_run_name
def run_name_to_experiment_definition(run_name: str) -> ExperimentDefinition:
    def __parse_modeltype(name: str) -> DUModel:
        name = name.replace("-", "_")
        if name in DUModel._member_names_:
            return DUModel[name]
        else:
            raise ValueError(f"Unknown DUModel: {name}")

    def __parse_distillmode(name: str) -> DistillMode:
        if name == "ft-teacher":
            return DistillMode.FINETUNE_TEACHER
        elif name == "ft-student":
            return DistillMode.FINETUNE_STUDENT
        elif name == "ft-student-newinit":
            return DistillMode.FINETUNE_STUDENT_NEWINIT
        elif name == "dt-student":
            return DistillMode.DISTILLATION
        elif name == "dt-student-1phase":
            return DistillMode.DISTILLATION_IN_1PHASE
        else:
            raise ValueError(f"Unknown mode identifier: {name}")

    def __parse_dataset_name(name: str) -> SUPPORTED_DATASET:
        ds: str
        name = name.replace("-chatgpt", "")
        for ds in get_args(SUPPORTED_DATASET):
            if ds.lower() == name:
                return ds

        raise ValueError(f"Unknown dataset identifier: {name}")

    def __parse_epochs(component: str) -> int:
        # 100epochs
        return int(component.replace("epochs", ""))

    def __parse_vocab(component: str):
        # tinyvocab smallvocab
        if component == "tinyvocab":
            return DEFAULTS.TINY_VOCAB
        elif component == "smallvocab":
            return DEFAULTS.SMALL_VOCAB
        else:
            raise ValueError(f"Unknown vocab: {component}")

    def __parse_layers(component: str) -> list[int]:
        # distill_funsd_layers-1-5-9_100epochs
        layers_str = component.replace("layers-", "")
        layers = layers_str.split("-")
        return [int(l) for l in layers]

    def __parse_distillation_loss_weights(
        component: str,
    ) -> tuple[float, float, float, float]:
        # alpha=1-beta=1-gamma=100-delta=0.01
        parts = component.split("-")
        alpha = float(parts[0].replace("alpha=", ""))
        beta = float(parts[1].replace("beta=", ""))
        gamma = float(parts[2].replace("gamma=", ""))
        delta = float(parts[3].replace("delta=", ""))
        return (alpha, beta, gamma, delta)

    # DEBUG
    is_teacher_in_eval_mode = False
    teacher_eval_str = "_teacher_eval"
    if teacher_eval_str in run_name:
        is_teacher_in_eval_mode = True
        run_name = run_name.replace(teacher_eval_str, "")

    # LayoutLMv3-TextAndImage_ft-student_docvqa_layers-2-5-8-11_30epochs
    components = run_name.split("_")
    assert len(components) >= 4

    # First component is always the model type
    model_type_str = components[0]
    model_type: DUModel = __parse_modeltype(model_type_str)

    # Second component is always the training-type / DistillMode
    distill_mode_str = components[1]
    distill_mode: DistillMode = __parse_distillmode(distill_mode_str)

    # Third componenent is always the dataset
    dataset_str = components[2]
    use_chatgpt_labels: bool = "-chatgpt" in dataset_str
    dataset: SUPPORTED_DATASET = __parse_dataset_name(dataset_str)

    # LayoutLMv3-TextAndImage_dt-student_sroie_layers-2-8_100epochs_alpha=1-beta=1-gamma=100-delta=0.01.json

    # Fourth componenent is where it gets tricky, may be vocab, layers, epoch, alpha-beta-gamma-delta or run_name_suffix
    num_epochs = None
    vocab = None
    layers = DEFAULTS.DEFAULT_LAYER_MAP
    (alpha, beta, gamma, delta) = (
        DEFAULTS.ALPHA,
        DEFAULTS.BETA,
        DEFAULTS.GAMMA,
        DEFAULTS.DELTA,
    )

    # TODO: run_name_suffix not supported yet
    # TODO: this breaks if the run_name_suffix contains epochs, vocab, layers or alpha
    use_ce_loss_for_distillation = False
    for component in components[3:]:
        if "epochs" in component:
            num_epochs = __parse_epochs(component)
        elif "vocab" in component:
            vocab = __parse_vocab(component)
        elif "layers" in component:
            layers = __parse_layers(component)
        elif (
            "alpha=" in component
            and "beta=" in component
            and "gamma=" in component
            and "delta=" in component
        ):
            (alpha, beta, gamma, delta) = __parse_distillation_loss_weights(component)
        elif "logits-distill-ce-loss" in component:
            use_ce_loss_for_distillation = True
        else:
            raise ValueError(f"Unknown component in run_name: {component}")

    return ExperimentDefinition(
        model=model_type,
        mode=distill_mode,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=tuple(layers),
        vocab_name=vocab,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        use_ce_loss_for_logits_distillation=use_ce_loss_for_distillation,
    )


if __name__ == "__main__":
    run_name = "LayoutLMv3-TextAndImage_dt-student_sroie_layers-2-8_100epochs_alpha=1-beta=1-gamma=100-delta=0.01"
    ed = run_name_to_experiment_definition(run_name)
    print(run_name)
    print(ed)
