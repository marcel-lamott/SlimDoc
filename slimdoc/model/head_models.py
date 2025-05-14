from typing import Union
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    LiltModel,
    LayoutLMv3Model,
    LiltConfig,
    LayoutLMv3Config,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from slimdoc import CONSTS, SUPPORTED_CONFIG, DUModel
from slimdoc.model.custom_lilt import CustomLiltModel


def get_base_model_pretrained_modelid(model_type: DUModel):
    if (
        model_type == DUModel.LiLT_TextFlow
        or model_type == DUModel.LiLT_TextAndLayoutFlow
        or model_type == DUModel.LiLT_TextFlow_NoLayout
    ):
        return CONSTS.LILT_HF_MODEL_ID
    elif (
        model_type == DUModel.LayoutLMv3_TextAndImage
        or model_type == DUModel.LayoutLMv3_TextOnly
        or model_type == DUModel.LayoutLMv3_NoVisionNoLayout
    ):
        return CONSTS.LAYOUTLMV3_HF_MODEL_ID
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_base_model(model_type: DUModel, config=None):
    if (
        model_type == DUModel.LiLT_TextFlow
        or model_type == DUModel.LiLT_TextFlow_NoLayout
    ):
        # Standard LiLT
        return LiltModel(config=config)
    elif model_type == DUModel.LiLT_TextAndLayoutFlow:
        # Customized version of LiLT also returning hidden states for layout flow
        return CustomLiltModel(config=config)
    elif (
        model_type == DUModel.LayoutLMv3_TextAndImage
        or model_type == DUModel.LayoutLMv3_TextOnly
        or model_type == DUModel.LayoutLMv3_NoVisionNoLayout
    ):
        return LayoutLMv3Model(config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class DUModelForTokenClassification(nn.Module):
    def __init__(
        self,
        model_type: DUModel,
        load_pretrained_hf: bool,
        num_labels: int,
        config: SUPPORTED_CONFIG = None,
    ):
        # assert du_model is not None or model_id is not None, 'Provide either a model instance or a HuggingFace model id'

        super(DUModelForTokenClassification, self).__init__()
        model_id = get_base_model_pretrained_modelid(model_type)
        self.num_labels = num_labels
        self.config = config or AutoConfig.from_pretrained(model_id)
        assert (
            self.config.num_hidden_layers <= 12
        ), f"Only models with at most 12 transformer layers are supported: {self.config.num_hidden_layers=}, {model_id=}"

        self.du_model = (
            AutoModel.from_pretrained(model_id, config=self.config)
            if load_pretrained_hf
            else get_base_model(model_type, self.config)
        )

        # Classification Head: Linear layer that predicts class labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.init_weights()

    def init_weights(self):
        # Initialize weights for the custom head
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        # The higher the embdding size the lower the std
        # Adrian: 0.02 might also be too much
        # TODO: run with glorot/he initialization https://pytorch.org/docs/stable/nn.init.html

    def forward(self, input_ids, bbox, attention_mask=None, **kwargs):
        outputs: BaseModelOutputWithPooling = self.du_model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, **kwargs
        )

        # shape: (batch_size, sequence_length, hidden_size)
        last_hidden_states: torch.Tensor = outputs.last_hidden_state
        x = self.dropout(last_hidden_states)
        logits: torch.Tensor = self.classifier(x)

        return {
            "logits": logits,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states,
        }


class DUModelForExtractiveQA(nn.Module):
    def __init__(
        self,
        model_type: DUModel,
        load_pretrained_hf: bool,
        config: SUPPORTED_CONFIG = None,
    ):
        # assert du_model is not None or model_id is not None, 'Provide either a model instance or a HuggingFace model id'

        super(DUModelForExtractiveQA, self).__init__()
        model_id = get_base_model_pretrained_modelid(model_type)
        self.config = config or AutoConfig.from_pretrained(model_id)
        assert (
            self.config.num_hidden_layers <= 12
        ), f"Only models with at most 12 transformer layers are supported: {self.config.num_hidden_layers=}, {model_id=}"

        self.du_model = (
            AutoModel.from_pretrained(model_id, config=self.config)
            if load_pretrained_hf
            else get_base_model(model_type, self.config)
        )

        # Classification Head: Linear layer that predicts class labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.start_logits = nn.Linear(self.config.hidden_size, 1)  # Start logits
        self.end_logits = nn.Linear(self.config.hidden_size, 1)  # End logits

        # Initialize weights for the QA head
        self.init_weights()

    def init_weights(self):
        # Initialize weights for the custom head
        nn.init.normal_(self.start_logits.weight, std=0.02)
        nn.init.zeros_(self.start_logits.bias)
        nn.init.normal_(self.end_logits.weight, std=0.02)
        nn.init.zeros_(self.end_logits.bias)

    def forward(self, input_ids, bbox, attention_mask=None, **kwargs):
        outputs: BaseModelOutputWithPooling = self.du_model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, **kwargs
        )

        # shape: (batch_size, sequence_length, hidden_size)
        last_hidden_states: torch.Tensor = outputs.last_hidden_state
        x = self.dropout(last_hidden_states)

        # Pass the sequence output through the QA head
        start_logits: torch.Tensor = self.start_logits(x).squeeze(
            -1
        )  # Shape: [batch_size, seq_len]
        end_logits: torch.Tensor = self.end_logits(x).squeeze(
            -1
        )  # Shape: [batch_size, seq_len]

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states,
        }
