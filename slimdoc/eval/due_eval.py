"""
Implements the evluation for DUE datasets (DocVQA, InfoVQA, WTQ)
"""

from argparse import Namespace
import argparse
import json
import pathlib
import shelve
from typing import Optional

import numpy as np
from slimdoc import ENV
import due_evaluator
import jsonlines


"""
Dataset	                    Metric	                                        Case insensitive
DocVQA, InfographicsVQA	    ANLS	                                        Yes
Kleister Charity, DeepForm	F1	                                            Yes
PapersWithCode	            GROUP-ANLS	                                    Yes
WikiTableQuestions	        WTQ	                                            No (handled by metric itself)
TabFact	                    F1 (obtained value will be equal to Accuracy)	No
"""

# Dataset names are the ones used for initialization of DUEDataset class
__dataset_to_metric_ignorecase: dict = {
    "DocVQA": ("ANLS", True),
    "InfographicsVQA": ("ANLS", True),
    "KleisterCharity": ("F1", True),
    "DeepForm": ("F1", True),
    "PWC": ("GROUP-ANLS", True),
    "WikiTableQuestions": ("WTQ", False),
    "TabFact": ("F1", False),
}


def __create_eval_file_data(
    dataset_name: str,
    split: str,
    predictions: dict[str, str],
    reference_path: pathlib.Path,
    only_our_samples: bool = False,
) -> list[dict]:
    lines: list[dict] = []

    reference = open(reference_path)
    # reference_dict = {x['name']: x for x in [json.loads(line) for line in reference]}
    reference_list = [json.loads(line) for line in reference]
    filtered_reference_list = (
        [] if only_our_samples else None
    )  # only relevant for parameter only_out_samples=True

    # for sample_id, pred in predictions.items():
    for ref in reference_list:
        sample_id = ref["name"]

        if only_our_samples:
            if not any([a for a in ref["annotations"] if a["id"] in predictions]):
                continue
            else:
                # Restrict ref to only our annotations
                ref["annotations"] = [
                    a for a in ref["annotations"] if a["id"] in predictions
                ]
                filtered_reference_list.append(ref)

        res = dict()
        res["name"] = sample_id
        res["language"] = ref["language"]
        res["split"] = split

        res["annotations"] = [
            {
                "id": a["id"],
                "key": a["key"],
                "values": [
                    {"value": predictions[a["id"]] if a["id"] in predictions else ""}
                ],
            }
            for a in ref["annotations"]
        ]

        for i, a in enumerate(ref["annotations"]):
            if "metadata" in a:
                res["annotations"][i]["metadata"] = a["metadata"]
        lines.append(res)

    return lines, filtered_reference_list


def __get_evaluation_results(
    reference_path: pathlib.Path,
    outfile_path: pathlib.Path,
    metric: str,
    ignore_case: bool,
    property_set: Optional[set[str]] = None,
    filtered_reference_list=None,
):
    out_file = open(outfile_path)
    reference = open(reference_path)

    reference = filtered_reference_list or [json.loads(line) for line in reference]
    predictions = [json.loads(line) for line in out_file]
    # predictions = list(reference)

    evaluator = due_evaluator.DueEvaluator(
        reference, predictions, property_set, ignore_case, out_file.name, metric
    )
    has_feature_scores = evaluator.general_scorer.support_feature_scores()
    # if has_feature_scores:
    #     results = {key: scorer.score() for key, scorer in evaluator.property_scorers.items()}
    # else:
    results = {"score": evaluator.general_scorer.score()}
    # print(f'REFERENCE: {len(reference)}\n{reference[0]}\nPREDICTIONS: {len(predictions)}\n{predictions[0]}\nRES: {results}, evaluator:{evaluator.property_scorers}')
    # print(f'has_feature_scores: {has_feature_scores}')

    return results


# Predictions ened to be dict questionId -> extracted answer span


def evaluate_due_results(
    dataset_name: str,
    split: str,
    predictions: dict[str, str],
    only_our_samples: bool = False,
):
    # Preparation
    metric, ignore_case = __dataset_to_metric_ignorecase[dataset_name]

    # Get GT file
    reference_document_jsonl_path: pathlib.Path = (
        ENV.DATASETS_DIR / dataset_name / "data" / split / "document.jsonl"
    )

    # write document.jsonl file with results
    eval_file_data, filtered_reference_list = __create_eval_file_data(
        dataset_name=dataset_name,
        split=split,
        predictions=predictions,
        reference_path=reference_document_jsonl_path,
        only_our_samples=only_our_samples,
    )

    # get temp filename
    results_document_jsonl_path: pathlib.Path = (
        ENV.TEMP_DIR / f"eval_{dataset_name}_{split}.jsonl"
    )
    # Delete jsonl file if already exists
    if results_document_jsonl_path.exists():
        results_document_jsonl_path.unlink()
    # Write data to jsonl file for evaluation
    with jsonlines.open(results_document_jsonl_path, mode="w") as writer:
        for x in eval_file_data:
            writer.write(x)

    results = __get_evaluation_results(
        reference_path=reference_document_jsonl_path,
        outfile_path=results_document_jsonl_path,
        metric=metric,
        ignore_case=ignore_case,
        filtered_reference_list=filtered_reference_list,
    )

    # print(f'{dsid}: {results}')

    return np.average(list(results.values()))


if __name__ == "__main__":
    p = ENV.DATASETS_DIR / "DUE" / "DeepForm" / "data" / "test" / "document.jsonl"
    # x = __call_eval_tool(p,p)
    # print(x)
    ds = "DeepForm"
    metric, ignore_case = __dataset_to_metric_ignorecase[ds]
    res = __get_evaluation_results(p, p, metric, ignore_case)
    print(res)
