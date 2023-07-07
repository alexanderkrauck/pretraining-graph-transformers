from utils.modeling_graphormer_improved import *

from torch import Tensor
from typing import Callable

import torch.nn.functional as F


def gaussian(x, mean, std):
    pi = 3.1415927410125732
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gaussian_size, hidden_size = config.gaussian_size, config.hidden_size
        self.means = nn.Embedding(1, self.gaussian_size)
        self.stds = nn.Embedding(1, self.gaussian_size)
        self.mul = nn.Linear(hidden_size, 1)  # NOTE: this is changed from the original
        self.bias = nn.Linear(hidden_size, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type_embeddings):
        mul = self.mul(edge_type_embeddings)
        bias = self.bias(edge_type_embeddings)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.gaussian_size)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.embed_dim = config.embedding_dim

        self.num_heads = config.num_attention_heads

        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.in_proj: Callable[[Tensor], Tensor] = nn.Linear(
            self.embed_dim, self.embed_dim * 3, bias=config.bias
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)

        self.attention_dropout_module = torch.nn.Dropout(
            p=config.attention_dropout, inplace=False
        )

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor = None,
    ) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) + attn_bias

        attn_probs = F.softmax(attn_weights, -1)
        attn_probs = self.attention_dropout_module(attn_probs)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        attn = self.out_proj(attn)
        return attn


class Graphormer3DGraphEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = config.embedding_dim
        self.num_attention_heads = config.num_attention_heads
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout
        self.q_noise = config.q_noise
        self.qn_block_size = config.qn_block_size
        self.pre_layernorm = config.pre_layernorm

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)

        self.activation_dropout_module = torch.nn.Dropout(
            p=config.activation_dropout, inplace=False
        )

        # Initialize blocks
        self.activation_fn = ACT2FN[config.activation_fn]
        self.self_attn = SelfMultiheadAttention(config)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.fc1 = self.build_fc(
            self.embedding_dim,
            config.ffn_embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )
        self.fc2 = self.build_fc(
            config.ffn_embedding_dim,
            self.embedding_dim,
            q_noise=config.q_noise,
            qn_block_size=config.qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_fc(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        input_nodes: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[
            torch.Tensor
        ] = None,  # NOTE: not sure if that can have any use in the attention
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes = self.self_attn(
            query=input_nodes,
            attn_bias=self_attn_bias,
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        return input_nodes


class Graphormer3DGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable

        self.input_graph_node_feature = GraphormerGraphNodeFeature(config)
        self.output_graph_node_feature = GraphormerGraphNodeFeature(config)
        self.graph_attn_bias = GraphormerGraphAttnBias(config)

        self.gbf = GaussianLayer(config)
        self.edge_proj = nn.Linear(config.gaussian_size, config.hidden_size)
        self.bias_proj = nn.Sequential(
            nn.Linear(config.gaussian_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_attention_heads),
        )
        self.clf_gbf_feature = nn.Parameter(
            torch.randn(config.gaussian_size), requires_grad=True
        )

        self.embed_scale = config.embed_scale

        if config.q_noise > 0:
            self.quant_noise = quant_noise(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                config.q_noise,
                config.qn_block_size,
            )
        else:
            self.quant_noise = None

        if config.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if config.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                Graphormer3DGraphEncoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if config.freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(config.num_trans_layers_to_freeze):
            m = self.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_nodes,
        pos,
        last_state_only: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        n_graph, n_node = input_nodes.size()[:2]
        padding_mask = (input_nodes[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        # compute 3D stuff
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        input_graph_node_embedding = self.input_graph_node_feature(input_nodes)

        edge_type_embeddings = input_graph_node_embedding[:, 1:].view(
            n_graph, n_node, 1, -1
        ) + self.output_graph_node_feature(input_nodes, add_graph_token=False).view(
            n_graph, 1, n_node, -1
        )
        gbf_features = self.gbf(dist, edge_type_embeddings)

        clf_gbf_repeated_input = self.clf_gbf_feature.unsqueeze(0).repeat(
            input_nodes.shape[0], input_nodes.shape[1], 1, 1
        )
        clf_gbf_repeated_output = self.clf_gbf_feature.unsqueeze(0).repeat(
            input_nodes.shape[0], 1, input_nodes.shape[1] + 1, 1
        )

        gbf_features = torch.cat([clf_gbf_repeated_input, gbf_features], dim=2)
        gbf_features = torch.cat([clf_gbf_repeated_output, gbf_features], dim=1)

        edge_features = gbf_features.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        graph_node_feature = input_graph_node_embedding
        +self.edge_proj(  # NOTE: maybe don't share the same embadding?
            edge_features.sum(dim=-2)
        )

        ##Main model

        attn_bias = self.bias_proj(gbf_features).permute(0, 3, 1, 2).contiguous()
        attn_bias.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_bias = attn_bias.view(-1, n_node + 1, n_node + 1)

        if self.emb_layer_norm is not None:
            graph_node_feature = self.emb_layer_norm(graph_node_feature)

        graph_node_feature = self.dropout_module(graph_node_feature)
        graph_node_feature = graph_node_feature.transpose(0, 1).contiguous()

        # Here the layers
        inner_states = []
        if not last_state_only:
            inner_states.append(graph_node_feature)

        for layer in self.layers:
            graph_node_feature = layer(
                graph_node_feature,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(graph_node_feature)

        graph_rep = graph_node_feature[0]

        if last_state_only:
            inner_states = [graph_node_feature]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep


class Graphormer3DModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    Graphormer3DForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_nodes = config.max_nodes

        self.graph_encoder = Graphormer3DGraphEncoder(config)

        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(config, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(
            config.embedding_dim, config.embedding_dim
        )
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        self.post_init()

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_nodes,
        pos,
        masked_tokens=None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inner_states, graph_rep = self.graph_encoder(
            input_nodes,
            pos,
        )

        # last inner state, then revert Batch and Graph len
        input_nodes = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        input_nodes = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(input_nodes))
        )

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            input_nodes = torch.nn.functional.linear(
                input_nodes, self.graph_encoder.embed_tokens.weight
            )

        if not return_dict:
            return tuple(x for x in [input_nodes, inner_states] if x is not None)
        return BaseModelOutputWithNoAttention(
            last_hidden_state=input_nodes, hidden_states=inner_states
        )

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes


class Graphormer3DForGraphClassification(GraphormerPreTrainedModel):
    """
    Implementation of the 3d Graphormer model for graph classification as in the repository of the paper.
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Graphormer3DModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.equal_nan_loss_weighting = config.equal_nan_loss_weighting
        self.classification_task = config.classification_task
        self.classifier_head = GraphormerDecoderHead(
            self.embedding_dim, self.num_classes
        )
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes,
        pos,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_nodes,
            pos,
            return_dict=True,
        )
        outputs, hidden_states = (
            encoder_outputs["last_hidden_state"],
            encoder_outputs["hidden_states"],
        )

        clf_outputs = outputs[:, 0]
        logits = self.classifier_head(clf_outputs).contiguous()

        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)

            if self.classification_task == "regression":  # regression
                if self.equal_nan_loss_weighting and not (
                    len(labels.shape) == 1 or labels.shape[-1] == 1
                ):
                    loss_fct = MSELoss(reduction="none")
                    loss = loss_fct(
                        logits[mask].squeeze(), labels[mask].squeeze().float()
                    )
                    n_not_nan = mask.squeeze().sum(1)
                    loss_weights = (torch.ones_like(logits) / n_not_nan.unsqueeze(1))[
                        mask
                    ]
                    loss = (loss * loss_weights).sum() / logits.shape[
                        0
                    ]  # better scaling for lr
                else:
                    loss_fct = MSELoss()
                    loss = loss_fct(
                        logits[mask].squeeze(), labels[mask].squeeze().float()
                    )
            elif self.classification_task == "classification" and (
                len(labels.shape) == 1 or labels.shape[-1] == 1
            ):  # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits[mask].view(-1, self.num_classes), labels[mask].view(-1)
                )
            else:  # Binary multi-task classification
                if self.equal_nan_loss_weighting:
                    loss_fct = BCEWithLogitsLoss(reduction="none")
                    loss = loss_fct(logits[mask], labels[mask])
                    n_not_nan = mask.sum(1)
                    loss_weights = (torch.ones_like(logits) / n_not_nan.unsqueeze(1))[
                        mask
                    ]
                    loss = (loss * loss_weights).sum() / logits.shape[
                        0
                    ]  # better scaling for lr
                else:
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=hidden_states, attentions=None
        )

class Graphormer3DForPretraining(
    GraphormerPreTrainedModel
):  # TODO check the GraphormerPreTrainedModel class and see if it is necessary to use it
    """
    This model can be used for pretraining the Graphormer model.

    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim

        self.pretraining_method = config.pretraining_method
        self.num_node_properties = config.num_node_properties
        self.num_edge_properties = config.num_edge_properties
        self.single_embedding_offset = config.single_embedding_offset
        self.reconstruction_method = config.reconstruction_method
        self.detach_target_embedding = config.detach_target_embedding

        if (
            self.pretraining_method == "mask_prediction"
        ):  # TODO: consider using an own embedding for each property of atoms. (might be better)
            self.mask_prob = config.mask_prob
            if self.reconstruction_method == "index_prediction":
                self.decoders = nn.ModuleList(
                    [
                        nn.Linear(self.embedding_dim, self.single_embedding_offset)
                        for _ in range(self.num_node_properties)
                    ]
                )

                self.loss = CrossEntropyLoss()

            elif self.reconstruction_method == "embedding_prediction":
                self.decoder = nn.Linear(self.embedding_dim, self.embedding_dim)
                self.loss = MSELoss()
                self.target_embedding = self.encoder.graph_encoder.graph_node_feature

        if self.pretraining_method == "noise_prediction":
            self.decoder = nn.Linear(self.embedding_dim, 1)
            self.loss = MSELoss()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes,
        pos,
        labels,
        mask,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_nodes,
            pos,
            return_dict=True,
        )
        outputs = encoder_outputs["last_hidden_state"]


        if self.pretraining_method == "mask_prediction":

            outputs = outputs[:, 1:]  # don't need the CLS token

            # NOTE: this currently removes the batch_dim. but it needs to happen because otherwise in the second dim the size might not be the same anymore.
            # However, the loss then is not averaged uniformly over the batch. This might be a problem.
            # Large graphs get a higher weight in the loss than small graphs.
            # I could just
            masked_outputs = outputs[mask]

            # Decode the masked input
            if self.reconstruction_method == "index_prediction":  # TODO: test this
                decoded_masked_outputs_logits = torch.stack(
                    [decoder(masked_outputs) for decoder in self.decoders], dim=1
                ).transpose(1,2)

                loss = cross_entropy(decoded_masked_outputs_logits, labels)

            elif self.reconstruction_method == "embedding_prediction":
                decoded_masked_outputs_logits = self.decoder(masked_outputs)
                
                embedded_target = self.target_embedding(
                        labels,
                        add_graph_token=False,
                    )
                
                if self.detach_target_embedding:
                    embedded_target = embedded_target.detach()

                loss = self.loss(
                    decoded_masked_outputs_logits,
                    embedded_target,
                ) #TODO: need regularization here, otherwise all embeddings will converge to a single value
        # NOTE: possibly assert that outputs on second dim are the same size as input_nodes on second dim

        elif self.pretraining_method == "noise_prediction":
            outputs = outputs[:, 1:][mask]  # don't need the CLS token

            predicted_noise = self.decoder(outputs) #we try to predict the noise magnitude. Then it is rotationally invariant
            loss = self.loss(predicted_noise, labels)


        return {
            "loss": loss,
            "outputs": outputs,
            "hidden_states": encoder_outputs["hidden_states"],
            "decoded_masked_outputs_logits": decoded_masked_outputs_logits,
        }


class Graphormer3DConfig(BetterGraphormerConfig):
    def __init__(self, gaussian_size: int = 128, **kwargs):
        super().__init__(**kwargs)

        self.gaussian_size = gaussian_size
