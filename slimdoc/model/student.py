from copy import deepcopy
from typing import Union

from transformers import AutoConfig, AutoModel, LiltModel, LayoutLMv3Model

from slimdoc import CONSTS, DUModel
from slimdoc.model.head_models import (
    DUModelForTokenClassification,
    DUModelForExtractiveQA,
)


def create_student(
    model_type: DUModel,
    teacher: Union[
        DUModelForTokenClassification, DUModelForExtractiveQA
    ],  # Teacher DU Model (in our custom Wrapper, i.e. DUModelForExtractiveQA or DUModelForTokenClassification)
    layer_map: dict[
        int, int
    ],  # map from which layers in the student to which layers in the teacher
    vocab_map,
):  # word 'the' is in col 40.000 in teachr vocab but in vocab 3 of student, maps these
    vocab_size = len(vocab_map) if vocab_map is not None else None

    # Create student config
    config = deepcopy(teacher.config)
    config.num_hidden_layers = len(layer_map)  # initially 12
    config.vocab_size = vocab_size or config.vocab_size  # initially 50265

    # Wrap our student base model into the correct wrapper module based on the teacher wrapper
    # Later we copy the weights from the teacher to the student
    if isinstance(teacher, DUModelForTokenClassification):
        student = DUModelForTokenClassification(
            model_type=model_type,
            load_pretrained_hf=False,
            num_labels=teacher.num_labels,
            config=config,
        )
    elif isinstance(teacher, DUModelForExtractiveQA):
        student = DUModelForExtractiveQA(
            model_type=model_type, load_pretrained_hf=False, config=config
        )
    else:
        raise ValueError(f"Unsupported teacher model type: {type(teacher).__name__}")

    t_state_dict = teacher.state_dict()

    # Temporarly delete embedding layer, otherwise we get size missmatch error
    if vocab_size is not None:
        # Name of the layer that has the word embeddings: du_model.embeddings.word_embeddings.weight
        embed_key = "du_model.embeddings.word_embeddings.weight"

        temp_state_dict = deepcopy(t_state_dict)
        del temp_state_dict[embed_key]
        student.load_state_dict(temp_state_dict, strict=False)
    else:
        student.load_state_dict(t_state_dict, strict=False)
    s_state_dict = student.state_dict()

    # copy each layer weights (check for state_dict with numbers)
    for s_layer_no, t_layer_no in enumerate(layer_map):
        if t_layer_no is None:
            continue
        s_layer_no = str(s_layer_no)
        t_layer_no = str(t_layer_no)
        for s_k in s_state_dict.keys():
            t_k = s_k.replace(s_layer_no, t_layer_no)
            # print(f's_k: {s_k}, t_k:{t_k}')
            if s_layer_no in s_k:
                s_state_dict[s_k] = t_state_dict[t_k]

    # copy embedding weights
    if vocab_size is not None:
        s_embed = s_state_dict[embed_key]
        t_embed = t_state_dict[embed_key]
        s_embed[: len(vocab_map)] = t_embed[vocab_map]
        s_state_dict[embed_key] = s_embed

    student.load_state_dict(s_state_dict, strict=True)
    # for k in student.state_dict():
    # 	print(k)

    return student


if __name__ == "__main__":
    # config = AutoConfig.from_pretrained(CONSTS.LILT_HF_MODEL_ID)
    # teacher = AutoModel.from_pretrained(CONSTS.LILT_HF_MODEL_ID, config=config)  # Load the LiLT backbone
    # teacher = LiLTForExtractiveQA()

    # layer_map = [2,5,8,11]
    # vocab_map = [1,2,3,4]
    # create_student(teacher=teacher,
    # 			layer_map=layer_map,
    # 			vocab_map=vocab_map)
    ...
