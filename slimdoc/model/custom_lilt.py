"""
The standard HuggingFace LiLT implementation only returns attentions and hidden states of the text flow.
Here we adopt the model to also return these values for the layout flow.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
from transformers import LiltModel, AutoConfig, AutoModel
from transformers.models.lilt.modeling_lilt import (
    LiltLayer,
    LiltEncoder,
    LiltSelfAttention,
    LiltAttention,
)
from transformers.utils import ModelOutput
import torch.nn as nn
import torch


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CustomLiltSelfAttention(LiltSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config=config, position_embedding_type=position_embedding_type)

    def forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        layout_value_layer = self.transpose_for_scores(
            self.layout_value(layout_inputs), r=self.channel_shrink_ratio
        )
        layout_key_layer = self.transpose_for_scores(
            self.layout_key(layout_inputs), r=self.channel_shrink_ratio
        )
        layout_query_layer = self.transpose_for_scores(
            self.layout_query(layout_inputs), r=self.channel_shrink_ratio
        )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        layout_attention_scores = torch.matmul(
            layout_query_layer, layout_key_layer.transpose(-1, -2)
        )

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        tmp_attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        tmp_layout_attention_scores = layout_attention_scores / math.sqrt(
            self.attention_head_size // self.channel_shrink_ratio
        )
        attention_scores = tmp_attention_scores + tmp_layout_attention_scores
        layout_attention_scores = tmp_layout_attention_scores + tmp_attention_scores

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            layout_attention_scores = layout_attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        layout_attention_probs = nn.Softmax(dim=-1)(layout_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        layout_attention_probs = self.dropout(layout_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            layout_attention_probs = layout_attention_probs * head_mask

        layout_context_layer = torch.matmul(layout_attention_probs, layout_value_layer)

        layout_context_layer = layout_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = layout_context_layer.size()[:-2] + (
            self.all_head_size // self.channel_shrink_ratio,
        )
        layout_context_layer = layout_context_layer.view(*new_context_layer_shape)
        # ---------
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        ##
        ## CHANGED HERE
        ##
        outputs = (
            (
                (context_layer, layout_context_layer),
                (attention_probs, layout_attention_probs),
            )
            if output_attentions
            else ((context_layer, layout_context_layer),)
        )

        return outputs


class CustomLiltAttention(LiltAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config=config, position_embedding_type=position_embedding_type)
        self.self = CustomLiltSelfAttention(
            config, position_embedding_type=position_embedding_type
        )


class CustomLiltLayer(LiltLayer):
    def __init__(self, config):
        super().__init__(config=config)
        self.attention = CustomLiltAttention(config)


class CustomLiltEncoder(LiltEncoder):
    # Copied from transformers.models.bert.modeling_bert.BertEncoder.__init__ with Bert->Lilt
    def __init__(self, config):
        super().__init__(config=config)
        self.layer = nn.ModuleList(
            [CustomLiltLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                ##
                ## CHANGED HERE
                ##
                all_hidden_states = all_hidden_states + (
                    (hidden_states, layout_inputs),
                )

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layout_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # layer_outputs = ((layer_output, layout_layer_output),) + outputs (outputs are only attention scores of text flow!)
                layer_outputs = layer_module(
                    hidden_states,
                    layout_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0][0]
            layout_inputs = layer_outputs[0][1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            ##
            ## CHANGED HERE
            ##
            all_hidden_states = all_hidden_states + ((hidden_states, layout_inputs),)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CustomLiltModel(LiltModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config=config, add_pooling_layer=add_pooling_layer)
        self.encoder = CustomLiltEncoder(config)


def get_custom_lilt():
    model_id = "SCUT-DLVCLab/lilt-roberta-en-base"
    config = AutoConfig.from_pretrained(model_id)
    du_model: CustomLiltModel = CustomLiltModel.from_pretrained(
        model_id, config=config
    )  # Load the LiLT backbone


"""
Layer Forward:
    def forward(
            self,
            hidden_states: torch.Tensor,
            layout_inputs: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
            self_attention_outputs = self.attention(
                hidden_states,
                layout_inputs,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0][0]
            layout_attention_output = self_attention_outputs[0][1]

            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
            layout_layer_output = apply_chunking_to_forward(
                self.layout_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layout_attention_output
            )
            outputs = ((layer_output, layout_layer_output),) + outputs

            return outputs


Lilt Attention Forward:
def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            layout_inputs,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0][0], hidden_states)
        layout_attention_output = self.layout_output(self_outputs[0][1], layout_inputs)
        outputs = ((attention_output, layout_attention_output),) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LiltSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


LiltSelfAttention forward output:
outputs = (
            ((context_layer, layout_context_layer), attention_probs)
            if output_attentions
            else ((context_layer, layout_context_layer),)
        )
"""
