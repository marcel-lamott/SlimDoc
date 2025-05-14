"""
Helperscript to run many experiments in series on a single device
"""

import argparse
import dataclasses
from itertools import chain
import itertools
import json
import time
from typing import Callable, Generator, Optional, Union, get_args

from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from slimdoc import ENV, DEFAULTS, SUPPORTED_DATASET, DUModel, DistillMode
from slimdoc.train import ExperimentDefinition, get_teacher_run_name
from slimdoc.train.train import main as train_main
from slimdoc.train.utils import get_device

# DUModel, datasets, use_chatgpt_labels, num_epochs
EXPERIMENT_DEFINITION_GENERATOR = Callable[
    [DUModel, list[SUPPORTED_DATASET], bool, int],
    Generator[ExperimentDefinition, None, None],
]

ALL_DATASETS_IDENTIFIER = "all_datasets"

"""
distill_{datasetsstr}_smallvocab_{num_epochs}epochs -> distill_{datasetsstr}_smallvocab_{2*num_epochs}epochs
"""


def get_ed_teacher(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    return ExperimentDefinition(
        model=model,
        mode=DistillMode.FINETUNE_TEACHER,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
    )


def create_experiment_definitions_distill_tinybert(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
        use_ce_loss_for_logits_distillation=True,
        alpha=1,
        beta=1,
        gamma=1,
        delta=1,
    )


def create_experiment_definitions_distill(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
    )


def create_experiment_definitions_minimum(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    yield get_ed_teacher(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )

    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
    )

    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.FINETUNE_STUDENT,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
    )


def create_experiment_definitions_vocab(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=DEFAULTS.SMALL_VOCAB,
    )

    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=DEFAULTS.TINY_VOCAB,
    )

    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.FINETUNE_STUDENT,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=DEFAULTS.SMALL_VOCAB,
    )

    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.FINETUNE_STUDENT,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=DEFAULTS.TINY_VOCAB,
    )


def create_experiment_definitions_layers1(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    for i in range(12):
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i]),
            vocab_name=None,
        )

        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i]),
            vocab_name=None,
        )


def create_experiment_definitions_layers1ft(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    for i in range(12):
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i]),
            vocab_name=None,
        )


def create_experiment_definitions_layers1dt(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    for i in range(12):
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i]),
            vocab_name=None,
        )


def create_experiment_definitions_layers2(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    for i in range(6):
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i, i + 6]),
            vocab_name=None,
        )

        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i, i + 6]),
            vocab_name=None,
        )


def create_experiment_definitions_layers3(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    for i in range(4):
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i, i + 4, i + 8]),
            vocab_name=None,
        )

        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple([i, i + 4, i + 8]),
            vocab_name=None,
        )


def create_experiment_definitions_layers4(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    for i in range(3):
        layer_map = tuple([i, i + 3, i + 6, i + 9])
        # we skip the default layers (done in collection 'minimum')
        if layer_map == DEFAULTS.DEFAULT_LAYER_MAP:
            continue

        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=layer_map,
            vocab_name=None,
        )

        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=layer_map,
            vocab_name=None,
        )


def create_experiment_definitions_outerlayers(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    layer_choices = [
        (0, 1, 2, 3),
        (8, 9, 10, 11),
        (0, 1, 2),
        (9, 10, 11),
        (0, 1),
        (10, 11),
    ]
    for layers in layer_choices:
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple(layers),
            vocab_name=None,
        )

        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple(layers),
            vocab_name=None,
        )


def create_experiment_definitions_outerlayersdt(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    layer_choices = [
        (0, 1, 2, 3),
        (8, 9, 10, 11),
        (0, 1, 2),
        (9, 10, 11),
        (0, 1),
        (10, 11),
    ]
    for layers in layer_choices:
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.DISTILLATION,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple(layers),
            vocab_name=None,
        )


def create_experiment_definitions_outerlayersft(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    layer_choices = [
        (0, 1, 2, 3),
        (8, 9, 10, 11),
        (0, 1, 2),
        (9, 10, 11),
        (0, 1),
        (10, 11),
    ]
    for layers in layer_choices:
        yield ExperimentDefinition(
            model=model,
            mode=DistillMode.FINETUNE_STUDENT,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
            student_layer_map=tuple(layers),
            vocab_name=None,
        )


def create_experiment_definitions_ablations(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION_IN_1PHASE,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
    )

    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.FINETUNE_STUDENT_NEWINIT,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=DEFAULTS.DEFAULT_LAYER_MAP,
        vocab_name=None,
    )


def create_experiment_definitions_debug(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    yield ExperimentDefinition(
        model=model,
        mode=DistillMode.DISTILLATION,
        dataset=dataset,
        use_chatgpt_labels=use_chatgpt_labels,
        student_layer_map=tuple([0]),
        vocab_name=None,
    )


def create_experiment_definitions_all(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    minimum_experiments = create_experiment_definitions_minimum(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    vocab_experiments = create_experiment_definitions_vocab(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    ablations_experiments = create_experiment_definitions_ablations(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers4_experiments = create_experiment_definitions_layers4(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers3_experiments = create_experiment_definitions_layers3(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers2_experiments = create_experiment_definitions_layers2(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers1_experiments = create_experiment_definitions_layers1(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    outerlayers_experiments = create_experiment_definitions_outerlayers(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    return chain(
        minimum_experiments,
        vocab_experiments,
        ablations_experiments,
        layers4_experiments,
        layers3_experiments,
        layers2_experiments,
        layers1_experiments,
        outerlayers_experiments,
    )


def create_experiment_definitions_all_except_layers(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    minimum_experiments = create_experiment_definitions_minimum(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    vocab_experiments = create_experiment_definitions_vocab(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    ablations_experiments = create_experiment_definitions_ablations(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    return chain(minimum_experiments, vocab_experiments, ablations_experiments)


def create_experiment_definitions_minimum_and_vocab(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    minimum_experiments = create_experiment_definitions_minimum(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    vocab_experiments = create_experiment_definitions_vocab(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    return chain(minimum_experiments, vocab_experiments)


def create_experiment_definitions_all_layers(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    # get teacher ED in case is not done yet
    teacher_eds = [
        get_ed_teacher(
            model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
        )
    ]
    layers4_experiments = create_experiment_definitions_layers4(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers3_experiments = create_experiment_definitions_layers3(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers2_experiments = create_experiment_definitions_layers2(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers1_experiments = create_experiment_definitions_layers1(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    outerlayers_experiments = create_experiment_definitions_outerlayers(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    return chain(
        teacher_eds,
        layers4_experiments,
        layers3_experiments,
        layers2_experiments,
        layers1_experiments,
        outerlayers_experiments,
    )


def create_experiment_definitions_all_layers_except1(
    model: DUModel, dataset: SUPPORTED_DATASET, use_chatgpt_labels: bool
):
    # get teacher ED in case is not done yet
    teacher_eds = [
        get_ed_teacher(
            model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
        )
    ]
    layers4_experiments = create_experiment_definitions_layers4(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers3_experiments = create_experiment_definitions_layers3(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    layers2_experiments = create_experiment_definitions_layers2(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    outerlayers_experiments = create_experiment_definitions_outerlayers(
        model=model, dataset=dataset, use_chatgpt_labels=use_chatgpt_labels
    )
    return chain(
        teacher_eds,
        layers4_experiments,
        layers3_experiments,
        layers2_experiments,
        outerlayers_experiments,
    )


# Maps series_name to (generator, description)
EXPERIMENT_SERIES: dict[str, (EXPERIMENT_DEFINITION_GENERATOR, str)] = {
    "distill": (
        create_experiment_definitions_distill,
        "Only distill student",
    ),
    "distill_tinybert": (
        create_experiment_definitions_distill_tinybert,
        "Distill similar to TinyBERT procedure (ce loss, all loss weights = 1)",
    ),
    "minimum": (
        create_experiment_definitions_minimum,
        "Train only teacher, distilled student and fine-tuned student",
    ),
    "vocab": (
        create_experiment_definitions_vocab,
        "Compare different vocab sizes for the student",
    ),
    "minimum_and_vocab": (
        create_experiment_definitions_minimum_and_vocab,
        "Train teacher, distilled student and fine-tuned student (default, small and tiny vocabs)",
    ),
    "ablations": (
        create_experiment_definitions_ablations,
        "Ablation studies: single phase distillation and new-init student finetuning",
    ),
    "all": (
        create_experiment_definitions_all,
        "Perform all experiments (teacher, distill, finetune, vocab-, layer- and ablation experiments)",
    ),
    "all_except_layers": (
        create_experiment_definitions_all_except_layers,
        "Perform all experiments except layer experiments (teacher, distill, finetune, vocab- and ablation experiments)",
    ),
    "all_layers": (
        create_experiment_definitions_all_layers,
        "Train teacher and all layer experiments (combines layers1, layers2, layers3, layers4 and outerlayers)",
    ),
    "all_layers_except1": (
        create_experiment_definitions_all_layers_except1,
        "Train teacher and all layer experiments (combines layers2, layers3, layers4 and outerlayers)",
    ),
    "layers1": (
        create_experiment_definitions_layers1,
        "Compare different choices of 1 layers for the student",
    ),
    "layers1ft": (
        create_experiment_definitions_layers1ft,
        "Compare different choices of 1 layers for the student (FT only)",
    ),
    "layers1dt": (
        create_experiment_definitions_layers1dt,
        "Compare different choices of 1 layers for the student (DT only)",
    ),
    "layers2": (
        create_experiment_definitions_layers2,
        "Compare different choices of 2 layers for the student",
    ),
    "layers3": (
        create_experiment_definitions_layers3,
        "Compare different choices of 3 layers for the student",
    ),
    "layers4": (
        create_experiment_definitions_layers4,
        "Compare different choices of 4 layers for the student",
    ),
    "outerlayers": (
        create_experiment_definitions_outerlayers,
        "Compare first and last N layers",
    ),
    "outerlayersdt": (
        create_experiment_definitions_outerlayersdt,
        "Compare first and last N layers (only DT)",
    ),
    "outerlayersft": (
        create_experiment_definitions_outerlayersft,
        "Compare first and last N layers (only FT)",
    ),
    "debug": (
        create_experiment_definitions_debug,
        "Distill a single student with 1 layer",
    ),
}


def _get_has_completed_experiment(ed: ExperimentDefinition):
    path = ENV.RUNNER_RESULTS_DIR / f"{ed.run_name}.json"
    # print(f'Check for runner results file at path: {path}')
    if path.exists():
        data = json.loads(path.read_text())
        return data["completed"]
    else:
        return False


def _set_has_completed_experiment(
    ed: ExperimentDefinition, has_completed: bool, error: Exception, elapsed: float
):
    path = ENV.RUNNER_RESULTS_DIR / f"{ed.run_name}.json"
    data = {
        "completed": has_completed,
        "error": str(error) if error else None,
        "elapsed": elapsed,
    }
    path.write_text(json.dumps(data))


def get_experiment_definitions(
    collection_name: str,
    model: DUModel,
    dataset: SUPPORTED_DATASET,
    use_chatgpt_labels: bool,
) -> list[ExperimentDefinition]:

    ed_generator_func, desc = EXPERIMENT_SERIES[collection_name]
    # DUModel, dataset, use_chatgpt_labels
    ed_generator = ed_generator_func(model, dataset, use_chatgpt_labels)
    experiments = list(ed_generator)

    return experiments


def get_all_datasets(use_chatgpt_labels: bool):
    if use_chatgpt_labels:
        return [
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
        ]


def get_experiment_definitions_for_all_datasets(
    collection_name: str, model: DUModel, use_chatgpt_labels: bool
) -> list[ExperimentDefinition]:
    all_ds = get_all_datasets(use_chatgpt_labels=use_chatgpt_labels)
    all_experiment_defs = []
    for dataset, _use_chatgpt_labels in all_ds:
        experiments: list[ExperimentDefinition] = get_experiment_definitions(
            collection_name=collection_name,
            model=model,
            dataset=dataset,
            use_chatgpt_labels=_use_chatgpt_labels,
        )
        all_experiment_defs.append(experiments)
    return list(itertools.chain(*all_experiment_defs))


def main(
    collection_name: str,
    model: DUModel,
    dataset: Union[SUPPORTED_DATASET, str],
    use_chatgpt_labels: bool,
    dataloader_num_workers: int,
    run_name_suffix: str,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    set_teacher_eval: bool,
    abort_on_error: bool,
    debug_run: bool,
):
    print(f"{alpha=} {beta=} {gamma=} {delta=}")

    if dataset == ALL_DATASETS_IDENTIFIER:
        experiments: list[ExperimentDefinition] = (
            get_experiment_definitions_for_all_datasets(
                collection_name=collection_name,
                model=model,
                use_chatgpt_labels=use_chatgpt_labels,
            )
        )
    else:
        experiments: list[ExperimentDefinition] = get_experiment_definitions(
            collection_name=collection_name,
            model=model,
            dataset=dataset,
            use_chatgpt_labels=use_chatgpt_labels,
        )

    # Prepare
    device = get_device()

    ed: ExperimentDefinition
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        "[yellow]({task.completed}/{task.total})",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        runner_task = progress.add_task(
            f"[red]Running {len(experiments)} experiments...", total=len(experiments)
        )

        for ed_template in experiments:
            # Update Experiment definition with passed arguments
            ed: ExperimentDefinition = dataclasses.replace(
                ed_template,
                run_name_suffix=run_name_suffix,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            progress.update(
                runner_task,
                description=f"[red]Running {len(experiments)} experiments: {ed.run_name}",
            )

            # skip completed experiments
            if _get_has_completed_experiment(ed) and not debug_run:
                print(f"Skipping already completed training: {ed.run_name}")
            else:
                try:
                    start_time = time.time()
                    train_main(
                        ed=ed,
                        dataloader_num_workers=dataloader_num_workers,
                        passed_device=device,
                        debug_run=debug_run,
                        progress=progress,
                        set_teacher_eval=set_teacher_eval,
                    )
                    _set_has_completed_experiment(
                        ed,
                        has_completed=True,
                        error=None,
                        elapsed=time.time() - start_time,
                    )
                except Exception as e:
                    print(f"Experiment {ed} failed: {e}")
                    _set_has_completed_experiment(
                        ed,
                        has_completed=False,
                        error=e,
                        elapsed=time.time() - start_time,
                    )
                    if abort_on_error:
                        raise e

            progress.update(runner_task, advance=1)

    print(f"FINISHED")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="SlimDoc Experiments Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ed_descriptions = ""
    for ed, (gen, desc) in EXPERIMENT_SERIES.items():
        ed_descriptions += f'-"{ed}": {desc}\n'
    parser.add_argument(
        "series_name",
        type=str,
        help=f"Speficy the name of the experiment-collection to run: {ed_descriptions}",
    )

    parser.add_argument(
        "model",
        type=int,
        help=f"Speficy a DU model via number: {[v for v in DUModel]}",
    )

    parser.add_argument(
        "dataset",
        type=str,
        # nargs='+',
        help=f"Speficy a dataset to use ({get_args(SUPPORTED_DATASET)}) or '{ALL_DATASETS_IDENTIFIER}', which runs the specified series and model for all datasets (depending on use_chatgpt_labels)",
    )

    parser.add_argument(
        "--use_chatgpt_labels",
        action="store_true",
        default=False,
        help=(
            "If set, use the ground truth established by the teacher LLM (ChatGPT). "
            f"By default, this is not set and the ground truth delivered with the datasets is used. If dataset={ALL_DATASETS_IDENTIFIER} only the datasets which have ChatGPT labels are used."
        ),
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=DEFAULTS.DATALOADER_NUM_WORKERS,
        help=f"Speficy the number of workers for parallel dataloading to speed up training. When set to 0, no parallelism is used and no new processes are spawned.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULTS.ALPHA,
        help=f"Distillation: Speficy the hidden state loss weight.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULTS.BETA,
        help=f"Distillation: Speficy the embedding loss weight.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULTS.GAMMA,
        help=f"Distillation: Speficy the attention loss weight.",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULTS.DELTA,
        help=f"Distillation: Speficy the logits loss weight.",
    )

    parser.add_argument(
        "--run_name_suffix",
        default="",
        help=(
            "If set, this string is appended to the run name. Allows the same experiment to be run multiple times."
        ),
    )

    parser.add_argument(
        "--abort_on_error",
        action="store_true",
        default=False,
        help=(
            "If set, execution is stopped if an error is encountered, otherwise the script continues with the next training."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "If set, perform a debug run. "
            "Experiments are still run even when they have been run before and no model checkpoint is saved."
        ),
    )

    parser.add_argument(
        "--set_teacher_eval",
        action="store_true",
        default=False,
        help=("If set, teacher model is set to 'eval' during distillation."),
    )

    # parser.add_argument(
    # 	"--num_epochs",
    # 	type=int,
    # 	default=None,
    # 	help=f"Speficy the number of epochs if you want to override the default settings (30 for DocVQA, InfoVQA, WTQ. 50 for SROIE, FUNSD. Double the epochs for distillation (due to 2 phases)).",
    # )

    args = parser.parse_args()
    print(args)

    # Assertions
    assert args.series_name in EXPERIMENT_SERIES
    assert (
        args.dataset in get_args(SUPPORTED_DATASET)
        or args.dataset == ALL_DATASETS_IDENTIFIER
    ), f"Unknown dataset: {args.dataset}"

    available_models = set(model.value for model in DUModel)
    assert (
        args.model in available_models
    ), f'Unknown model: "{args.model}", available models: {[m for m in DUModel]}'
    args.model = DUModel(args.model)

    assert args.dataloader_num_workers >= 0, "dataloader_num_workers must be >= 0"

    return args


if __name__ == "__main__":
    args = _parse_args()
    main(
        collection_name=args.series_name,
        model=args.model,
        dataset=args.dataset,
        use_chatgpt_labels=args.use_chatgpt_labels,
        dataloader_num_workers=args.dataloader_num_workers,
        run_name_suffix=args.run_name_suffix,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        set_teacher_eval=args.set_teacher_eval,
        abort_on_error=args.abort_on_error,
        debug_run=args.debug,
    )
