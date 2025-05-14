import argparse
import json
from slimdoc import DEFAULTS, ENV, DUModel, DistillMode
from slimdoc.eval.eval import evaluate_model
from slimdoc.eval.utils import get_all_required_experiments, get_eds_for_all_datasets
from slimdoc.train import ExperimentDefinition
from slimdoc.train.runner import (
    _get_has_completed_experiment,
    get_experiment_definitions,
)
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


def collect_finished_runs():
    all_runs = dict()
    run_name_to_ed = dict()
    for model, collection, include_chatgpt_labels in get_all_required_experiments():
        eds = get_eds_for_all_datasets(
            collection_name=collection,
            model=model,
            include_chatgpt_labels=include_chatgpt_labels,
        )
        for ed in eds:
            ed: ExperimentDefinition
            is_completed = _get_has_completed_experiment(ed)
            all_runs[ed.run_name] = is_completed
            run_name_to_ed[ed.run_name] = ed
            if ed.mode == DistillMode.DISTILLATION_IN_1PHASE:
                print(f'{ed.run_name} completed: {is_completed}')

    finished_runs = []
    finished_count = sum(all_runs.values())
    missing_count = len(all_runs) - finished_count
    print(f"{missing_count} missing runs:")
    for run_name, is_finished in sorted(all_runs.items(), key=lambda x: x[0]):
        if is_finished:
            finished_runs.append(run_name)
        else:
            ...
            #print(run_name)

    assert finished_count == len(finished_runs)
    print(f"{finished_count} of {len(all_runs)} models found.")
    return finished_runs, run_name_to_ed


def eval_all(
    ignore_existing: bool = False,
    complete_due_datasets: bool = False,
    split: str = "test",
    batch_size: int = 16,
    dataloader_num_workers: int = DEFAULTS.DATALOADER_NUM_WORKERS,
):
    finished_runs: list[str]
    run_name_to_ed: dict[str, ExperimentDefinition]
    finished_runs, run_name_to_ed = collect_finished_runs()

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        "[yellow]({task.completed}/{task.total})",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        runner_task = progress.add_task(
            f"[red]Evaluating {len(finished_runs)} models...", total=len(finished_runs)
        )

        skipped: int = 0
        for run_name in finished_runs:
            results_fpath = ENV.EVAL_RESULTS_DIR / f"{run_name}.json"

            #ed: ExperimentDefinition = run_name_to_ed[run_name]
            # if ed.mode == DistillMode.DISTILLATION_IN_1PHASE:
            #     print(f'1Phase-Distill! {ed.run_name}')
            #     print(f'exists: {results_fpath.exists()} at {results_fpath}')
            #     input()

            if results_fpath.exists() and not ignore_existing:
                #print(f"Skipping already evaluted model: {run_name}")
                skipped += 1
                progress.update(runner_task, advance=1)
                continue

            # Evaluate model
            score = evaluate_model(
                run_name=run_name,
                split=split,
                batch_size=batch_size,
                complete_due_datasets=complete_due_datasets,
                dataloader_num_workers=dataloader_num_workers,
                progress=progress,
            )

            # Export score
            with results_fpath.open("w") as f:
                data = {
                    "score": score,
                    "split": split,
                    "complete_due_datasets": complete_due_datasets,
                }
                json.dump(data, f)

            progress.update(runner_task, advance=1)
    
    print(f"Skipped {skipped} already evaluted models")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="SlimDoc Evaluation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ignore_existing",
        action="store_true",
        default=False,
        help="If set, already existing evaluation results are ignored and the model is evaluated again.",
    )

    parser.add_argument(
        "--complete_due_datasets",
        action="store_true",
        default=False,
        help="If set, we don't restrict evaluation of DUE datasets to our filtered subset but instead use the whole dataset.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=f"Speficy the split of the dataset on which to evaluate. (The dataset is infered automatically from the run_name)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help=f"Speficy the batch size."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=DEFAULTS.DATALOADER_NUM_WORKERS,
        help=f"Speficy the number of workers for parallel dataloading to speed up training. When set to 0, no parallelism is used and no new processes are spawned.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    eval_all(
        args.ignore_existing,
        args.complete_due_datasets,
        args.split,
        args.batch_size,
        args.dataloader_num_workers,
    )
