import json
import pathlib
from slimdoc import ENV


def __try_parse_json(myjson):
    try:
        json_object = json.loads(myjson)
        return json_object, True
    except ValueError as e:
        return None, False


def read_chatgpt_labels(dataset_name: str, split: str) -> dict[str, any]:
    # Read ChatGPT generated labels if applicable
    chatgpt_labels_dir: pathlib.Path = (
        ENV.DATASETS_DIR
        / "lapdoc"
        / "raw"
        / dataset_name.upper()
        / split
        / "ASCIIFormattingVerbalizer"
        / "results"
    )
    assert (
        chatgpt_labels_dir.exists()
    ), f"No ChatGPT generated labels found for dataset {dataset_name} split {split} at path: {chatgpt_labels_dir}"
    docid_to_chatgpt_labels = dict()
    for fpath in chatgpt_labels_dir.iterdir():
        chatgpt_json = fpath.read_text(encoding="utf-8")
        annotation_nr_to_label, valid_json = __try_parse_json(chatgpt_json)
        # The filename is the document ID
        # Questions of document are iterated in order
        if valid_json:
            docid_to_chatgpt_labels[fpath.stem] = annotation_nr_to_label

    return docid_to_chatgpt_labels


# small changes because questions in WebSRC were grouped
def read_chatgpt_labels_websrc() -> dict[str, any]:
    # Read ChatGPT generated labels if applicable
    chatgpt_labels_dir: pathlib.Path = (
        ENV.DATASETS_DIR
        / "lapdoc"
        / "raw"
        / "WEBSRC"
        / "train"
        / "ASCIIFormattingVerbalizer"
        / "results"
    )
    assert (
        chatgpt_labels_dir.exists()
    ), f"No ChatGPT generated labels found for dataset WebSRC split train at path: {chatgpt_labels_dir}"
    qid_to_chatgpt_labels = dict()
    for fpath in chatgpt_labels_dir.iterdir():
        chatgpt_json = fpath.read_text()
        annotation_nr_to_label, valid_json = __try_parse_json(chatgpt_json)
        # The filenames are comma-seperated Question IDS
        qids = fpath.stem.split(",")

        # Questions of document are iterated in order
        if valid_json:
            for i, qid in enumerate(qids):
                label_val = __get_value(annotation_nr_to_label, str(i))
                if label_val:
                    qid_to_chatgpt_labels[qid] = label_val

    return qid_to_chatgpt_labels


def get_label_value(docid_to_labels: dict, doc_id: str, key: str) -> str:
    # Use ChatGPT generated labels
    if doc_id not in docid_to_labels:
        # ChatGPT returned invalid json..
        return None
    else:
        annotation_nr_to_label = docid_to_labels[doc_id]
        return __get_value(annotation_nr_to_label, key)


def __get_value(results_dict, key):
    if key not in results_dict:
        return None
    else:
        value = results_dict[key]
        if value is None:
            return None
        elif isinstance(value, float) or isinstance(value, int):
            # Numbers we convert to string
            return str(value)
        if not isinstance(value, str):
            # Else we dont convert value to string because most of the time its
            # lists or dicts that were returned by ChatGPT
            return None

        return value
