# TODO write docstring

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import cross_entropy


from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.models.graphormer.modeling_graphormer import GraphormerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "graphormer-base-pcqm4mv1"
_CONFIG_FOR_DOC = "GraphormerConfig"


GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "clefourrier/graphormer-base-pcqm4mv1",
    "clefourrier/graphormer-base-pcqm4mv2",
    # See all Graphormer models at https://huggingface.co/models?filter=graphormer
]


def quant_noise(module, p, block_size):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/quant_noise.py

    Wraps modules and applies quantization noise to the weights for subsequent quantization with Iterative Product
    Quantization as described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights, see "And the Bit Goes Down:
          Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper which consists in randomly dropping
          blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
        raise NotImplementedError("Module unsupported for quant_noise.")

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        if module.weight.size(1) % block_size != 0:
            raise AssertionError("Input features must be a multiple of block sizes")

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            if module.in_channels % block_size != 0:
                raise AssertionError("Input channels must be a multiple of block sizes")
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            if k % block_size != 0:
                raise AssertionError("Kernel size must be a multiple of block size")

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class LayerDropModuleList(nn.ModuleList):
    """
    From:
    https://github.com/facebookresearch/fairseq/blob/dd0079bde7f678b0cd0715cbd0ae68d661b7226d/fairseq/modules/layer_drop.py
    A LayerDrop implementation based on [`torch.nn.ModuleList`]. LayerDrop as described in
    https://arxiv.org/abs/1909.11556.

    We refresh the choice of which layers to drop every time we iterate over the LayerDropModuleList instance. During
    evaluation we always iterate over all layers.

    Usage:

    ```python
    layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
    for layer in layers:  # this might iterate over layers 1 and 3
        x = layer(x)
    for layer in layers:  # this might iterate over all layers
        x = layer(x)
    for layer in layers:  # this might not iterate over any layers
        x = layer(x)
    ```

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class GraphormerGraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_atoms = config.num_atoms

        self.atom_encoder = nn.Embedding(
            config.num_atoms + 1, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.graph_token = nn.Embedding(1, config.hidden_size)

    def forward(self, input_nodes, add_graph_token=True):
        node_feature = (  # node feature + graph token
            self.atom_encoder(input_nodes).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        )
        if add_graph_token:
            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(
                input_nodes.shape[0], 1, 1
            )

            node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return node_feature


class GraphormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist

        # We do not change edge feature embedding learning, as edge embeddings are represented as a combination of the original features
        # + shortest path
        self.edge_encoder = nn.Embedding(
            config.num_edges + 1, config.num_attention_heads, padding_idx=0
        )

        self.edge_type = config.edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                config.num_edge_dis
                * config.num_attention_heads
                * config.num_attention_heads,
                1,
            )

        self.spatial_pos_encoder = nn.Embedding(
            config.num_spatial, config.num_attention_heads, padding_idx=0
        )

        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    def forward(self, input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type):
        n_graph, n_node = input_nodes.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()

            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, input_nodes > 1 to input_nodes - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, : self.multi_hop_max_dist, :]
            
            # [n_graph, n_node, n_node, max_dist, n_head]
            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist],
            )
            input_edges = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            input_edges = (
                input_edges.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_edge_features] -> [n_graph, n_head, n_node, n_node]
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges

        return graph_attn_bias


class GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = (
            self.kdim == config.embedding_dim and self.vdim == config.embedding_dim
        )

        self.num_heads = config.num_attention_heads

        self.attention_dropout = config.attention_dropout
        self.attention_dropout_module = torch.nn.Dropout(
            p=self.attention_dropout, inplace=False
        )

        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not (self.head_dim * config.num_attention_heads == self.embedding_dim):
            raise AssertionError("The embedding_dim must be divisible by num_heads.")
        self.scaling = self.head_dim**-0.5

        self.self_attention = True  # config.self_attention
        if not (self.self_attention):
            raise NotImplementedError(
                "The Graphormer model only supports self attention for now."
            )
        if self.self_attention and not self.qkv_same_dim:
            raise AssertionError(
                "Self-attention requires query, key and value to be of the same size."
            )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )
        self.q_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.out_proj = quant_noise(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias),
            config.q_noise,
            config.qn_block_size,
        )

        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len
        if not (embedding_dim == self.embedding_dim):
            raise AssertionError(
                f"The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim"
                f" {self.embedding_dim}."
            )
        if not (list(query.size()) == [tgt_len, bsz, embedding_dim]):
            raise AssertionError(
                "Query size incorrect in Graphormer, compared to model dimensions."
            )

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if (
                    (key_bsz != bsz)
                    or (value is None)
                    or not (src_len, bsz == value.shape[:2])
                ):
                    raise AssertionError(
                        "The batch shape does not match the key or value shapes provided to the attention."
                    )

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if (k is None) or not (k.size(1) == src_len):
            raise AssertionError(
                "The shape of the key generated in the attention is incorrect"
            )

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len:
                raise AssertionError(
                    "The shape of the generated padding mask for the key does not match expected dimensions."
                )
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError(
                "The attention weights generated do not match the expected dimensions."
            )

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)

        if v is None:
            raise AssertionError("No value generated")
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError(
                "The attention generated do not match the expected dimensions."
            )

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = (
                attn_weights_float.contiguous()
                .view(bsz, self.num_heads, tgt_len, src_len)
                .transpose(1, 0)
            )
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class GraphormerGraphEncoderLayer(nn.Module):
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
        self.self_attn = GraphormerMultiheadAttention(config)

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
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """
        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
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

        return input_nodes, attn


class GraphormerGraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout_module = torch.nn.Dropout(p=config.dropout, inplace=False)
        self.layerdrop = config.layerdrop
        self.embedding_dim = config.embedding_dim
        self.apply_graphormer_init = config.apply_graphormer_init
        self.traceable = config.traceable

        self.graph_node_feature = GraphormerGraphNodeFeature(config)
        self.graph_attn_bias = GraphormerGraphAttnBias(config)

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
                GraphormerGraphEncoderLayer(config)
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
        input_edges,
        attn_bias,
        spatial_pos,
        attn_edge_type,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        data_x = input_nodes
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        attn_bias = self.graph_attn_bias(
            input_nodes, attn_bias, spatial_pos, input_edges, attn_edge_type
        )

        if token_embeddings is not None:
            input_nodes = token_embeddings
        else:
            input_nodes = self.graph_node_feature(input_nodes)

        if perturb is not None:
            input_nodes[:, 1:, :] += perturb

        if self.embed_scale is not None:
            input_nodes = input_nodes * self.embed_scale

        if self.quant_noise is not None:
            input_nodes = self.quant_noise(input_nodes)

        if self.emb_layer_norm is not None:
            input_nodes = self.emb_layer_norm(input_nodes)

        input_nodes = self.dropout_module(input_nodes)

        input_nodes = input_nodes.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(input_nodes)

        for layer in self.layers:
            input_nodes, _ = layer(
                input_nodes,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(input_nodes)

        graph_rep = input_nodes[0]

        if last_state_only:
            inner_states = [input_nodes]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep


class GraphormerDecoderHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        """num_classes should be 1 for regression, or the number of classes for classification"""
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.num_classes = num_classes

    def forward(self, input_nodes, **unused):
        input_nodes = self.classifier(input_nodes)
        input_nodes = input_nodes + self.lm_output_learned_bias
        return input_nodes


class GraphormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GraphormerConfig
    base_model_prefix = "graphormer"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    main_input_name_nodes = "input_nodes"
    main_input_name_edges = "input_edges"

    def normal_(self, data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    def init_graphormer_params(self, module):
        """
        Initialize the weights specific to the Graphormer Model.
        """
        if isinstance(module, nn.Linear):
            self.normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            self.normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, GraphormerMultiheadAttention):
            self.normal_(module.q_proj.weight.data)
            self.normal_(module.k_proj.weight.data)
            self.normal_(module.v_proj.weight.data)

    def _init_weights(self, module):
        """
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # We might be missing part of the Linear init, dependant on the layer num
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GraphormerMultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphormerGraphEncoder):
            if module.apply_graphormer_init:
                module.apply(self.init_graphormer_params)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GraphormerModel):
            module.gradient_checkpointing = value


class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_nodes = config.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(config)

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
        input_edges,
        attn_bias,
        spatial_pos,
        attn_edge_type,
        perturb=None,
        masked_tokens=None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        inner_states, graph_rep = self.graph_encoder(
            input_nodes,
            input_edges,
            attn_bias,
            spatial_pos,
            attn_edge_type,
            perturb=perturb,
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


class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier_head = GraphormerDecoderHead(
            self.embedding_dim, self.num_classes
        )
        self.is_encoder_decoder = True

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes,
        input_edges,
        attn_bias,
        spatial_pos,
        attn_edge_type,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_nodes,
            input_edges,
            attn_bias,
            spatial_pos,
            attn_edge_type,
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

            if self.num_classes == 1:  # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif (
                self.num_classes > 1 and len(labels.shape) == 1
            ):  # One task classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits[mask].view(-1, self.num_classes), labels[mask].view(-1)
                )
            else:  # Binary multi-task classification #TODO: consider making multiple options for this
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits[mask], labels[mask])
                n_not_nan = mask.sum(1)
                loss_weights = (torch.ones_like(input) / n_not_nan.unsqueeze(1))[mask]
                loss = (loss * loss_weights).mean() * logits.shape[1] #better scaling for lr

        if not return_dict:
            return tuple(x for x in [loss, logits, hidden_states] if x is not None)
        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=hidden_states, attentions=None
        )


class GraphormerForPretraining(
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

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_nodes,
        input_edges,
        attn_bias,
        spatial_pos,
        attn_edge_type,
        n_nodes,
        labels,
        n_masked_nodes,
        mask,
        return_dict: Optional[bool] = None,
        **unused,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.pretraining_method == "mask_prediction":
            # Mask the input

            encoder_outputs = self.encoder(
                input_nodes,
                input_edges,
                attn_bias,
                spatial_pos,
                attn_edge_type,
                return_dict=True,
            )
            outputs = encoder_outputs["last_hidden_state"]
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
        return {
            "loss": loss,
            "outputs": outputs,
            "hidden_states": encoder_outputs["hidden_states"],
        }


class BetterGraphormerConfig(GraphormerConfig):
    def __init__(
        self,
        activation_dropout: float = 0.0,
        num_node_properties: int = 9,
        num_edge_properties: int = 3,
        single_embedding_offset: int = 512,
        pretraining_method: Optional[str] = None,
        mask_prob: Optional[float] = None,
        reconstruction_method: Optional[str] = None,
        detach_target_embedding: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
        ----
        activation_dropout: float
            The dropout probability for the activation function. Used in each Graphormer Encoder Layer between the 2 linear layers.
        num_node_properties: int
            The number of node properties.
        num_edge_properties: int
            The number of edge properties.
        single_embedding_offset: int
            The offset used for the embedding of each node property.
        pretraining_method: str
            The method used for pretraining the model. Currently only "mask_prediction" is supported.
        mask_prob: float
            The probability of masking a node property.
        reconstruction_method: str
            The method used for reconstructing the masked node properties. Currently only "index_prediction" and "embedding_prediction" are supported.
        detach_target_embedding: bool
            Whether to detach the target embedding from the gradient-computation graph. This forces the model to learn the embedding from the input. 
            
        All other arguments are passed to the GraphormerConfig superclass.
        """

        super().__init__(**kwargs)

        self.pretraining_method = pretraining_method
        self.num_node_properties = num_node_properties
        self.num_edge_properties = num_edge_properties
        self.single_embedding_offset = single_embedding_offset
        self.mask_prob = mask_prob
        self.reconstruction_method = reconstruction_method
        self.activation_dropout = activation_dropout
        self.detach_target_embedding = detach_target_embedding
