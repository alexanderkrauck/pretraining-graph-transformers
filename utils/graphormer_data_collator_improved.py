# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from transformers.utils import is_cython_available, requires_backends


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from transformers.models.graphormer import algos_graphormer  # noqa E402


# This is needed because in the model the same embedding is used for each of the dimensions atom/edges.
# An offset is needed to distinguish between them. However, this is really inefficient.
def convert_to_single_emb(x: np.ndarray, offset: int = 512):
    """
    Add an offset to each column/features in the input array to make them unique between columns.

    This is needed because in the model the same embedding is used for each of the dimensions atom/edges.
    An offset is needed to distinguish between them. However, this is really inefficient.

    Args:
    -----
    x : np.ndarray
        The input array.
    offset : int
        The offset to use.
    """
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def preprocess_item(item, num_edge_features: int = 3):
    """
    Preprocess a single item from a dataset.

    Args:
        item (Dict[str, Any]): A single item from a dataset.
        num_edge_features (int): The number of edge features to use. NOTE: This is mainly here for the speical case of graphs without any edges.
    """

    requires_backends(preprocess_item, ["cython"])

    # Transfer the data from the item to numpy arrays.
    # below assumes that the input has edge_attributes and node_features
    edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    num_nodes = item["num_nodes"]

    # Create node features with an offset.
    input_nodes = convert_to_single_emb(node_feature) + 1

    if num_edge_features != 1 and edge_attr.shape[-1] != num_edge_features:
        edge_attr = np.zeros([0, num_edge_features], dtype=np.int64)
    elif len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]

    # Create dense edge_attr matrix with an offset.
    attn_edge_type = np.zeros([num_nodes, num_nodes, num_edge_features], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # Binary dense adjacency matrix (true if edge exists).
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    # Compute shortest path and max distance.
    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    # NOTE: for now ignore unconnected nodes
    reachable_mask = shortest_path_result < 510
    max_dist = np.amax(shortest_path_result[reachable_mask])

    # input edges is of shape [num_nodes, num_nodes, max_dist, num_edge_features]
    # If there is a unconnected node in the graph, the input_edges will be [num_nodes, num_nodes, 510, num_edge_features]
    # That is quite large and I need to make sure the data is correctly preprocessed so this only happens for a few samples if at all.
    if max_dist != 0:
        input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    else:
        input_edges = np.zeros(
            [num_nodes, num_nodes, 0, num_edge_features], dtype=np.int32
        )
    attn_bias = np.zeros(
        [num_nodes + 1, num_nodes + 1], dtype=np.single
    )  # with graph token

    # Create Dictionary entries with all the data.
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
    item[
        "attn_bias"
    ] = attn_bias  # NOTE: As far as I can tell, this feature is completely useless
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = (
        shortest_path_result.astype(np.int64) + 1
    )  # we shift all indices by one for padding
    item["in_degree"] = (
        np.sum(adj, axis=1).reshape(-1) + 1
    )  # we shift all indices by one for padding
    item["out_degree"] = item["in_degree"]  # for undirected graph
    item["input_edges"] = input_edges + 1  # we shift all indices by one for padding

    # NOTE: Graphormer expects the target to be called "labels".
    # This does not make much sense to me, but I, for now, don't want to rewrite the model.
    if "target" in item:
        item["labels"] = item["target"]

    return item


class GraphormerDataCollator:
    def __init__(
        self, spatial_pos_max=20, num_edge_features=3, on_the_fly_processing=True
    ):
        """
        Data collator for Graphormer.

        Args:
        -----
        spatial_pos_max : int
            The maximum spatial position to use.
        num_edge_features : int
            The number of edge features that are assumed.
        on_the_fly_processing : bool
            If true, the preprocessing is done on the fly. If false, the data is expected to be processed already.
            If True, the TrainingArguments need to have the following parameters set: remove_unused_columns = False, num_edge_features = num_edge_features
        """

        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        self.spatial_pos_max = spatial_pos_max
        self.num_edge_features = num_edge_features
        self.on_the_fly_processing = on_the_fly_processing

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        if self.on_the_fly_processing:
            features = [preprocess_item(f, self.num_edge_features) for f in features]

        batch = {}

        # NOTE: per preprocessing edge_feat_size and edge_input_size is always the same.
        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in features)
        # TODO maybe remove this if check
        if self.num_edge_features is None:
            num_edge_features = features[0]["input_edges"].shape[-1]
        else:
            num_edge_features = self.num_edge_features
        batch_size = len(features)

        # Here things are scaled to the maximum size of the batch as I did before with OHG.
        # This kind of makes parts of the preprocessing useless.
        # Set attention bias to zero for all nodes scaled to maximum size + 1.
        batch["attn_bias"] = torch.zeros(
            batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float
        )

        # Create tensor for edge attribute (type) data
        batch["attn_edge_type"] = torch.zeros(
            batch_size, max_node_num, max_node_num, num_edge_features, dtype=torch.long
        )
        # Create tensor for spatial position data (shortest path distance)
        batch["spatial_pos"] = torch.zeros(
            batch_size, max_node_num, max_node_num, dtype=torch.long
        )
        batch["in_degree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        # Create tensor for node feature data
        batch["input_nodes"] = torch.zeros(
            batch_size, max_node_num, node_feat_size, dtype=torch.long
        )
        # NOTE: Here again input edges which is huge. not sure yet why.
        batch["input_edges"] = torch.zeros(
            batch_size,
            max_node_num,
            max_node_num,
            max_dist,
            num_edge_features,
            dtype=torch.long,
        )

        for ix, f in enumerate(features):
            for k in [
                "attn_bias",
                "attn_edge_type",
                "spatial_pos",
                "in_degree",
                "input_nodes",
                "input_edges",
            ]:
                f[k] = torch.tensor(f[k])

            n_nodes = f["input_nodes"].shape[0]
            max_dist = f["input_edges"].shape[2]

            above_max = f["spatial_pos"] >= self.spatial_pos_max
            if torch.sum(above_max) > 0:
                # so all that are above the max are set to -inf #TODO: f["attn_bias"] is completely useless. you can just use the spatial pos
                f["attn_bias"][1:, 1:][above_max] = float("-inf")
            # TODO: this is all square matrices so we could just use one dim

            batch["attn_bias"][ix, : n_nodes + 1, : n_nodes + 1] = f["attn_bias"]
            batch["attn_edge_type"][ix, :n_nodes, :n_nodes] = f["attn_edge_type"]
            batch["spatial_pos"][ix, :n_nodes, :n_nodes] = f["spatial_pos"]
            batch["in_degree"][ix, :n_nodes] = f["in_degree"]
            batch["input_nodes"][ix, :n_nodes] = f["input_nodes"]
            # TODO: this if check sorts out graphs without any edges that are in bad format. Maybe remove later.
            if (
                batch["input_edges"][ix, :n_nodes, :n_nodes, :max_dist].shape
                == f["input_edges"].shape
            ):
                batch["input_edges"][ix, :n_nodes, :n_nodes, :max_dist] = f[
                    "input_edges"
                ]

        batch["out_degree"] = batch["in_degree"]  # NOTE: for undirected graph only

        # Only add labels if they are in the features. For inference, or pretraining, the features won't have labels.
        if "labels" in features[0].keys():
            sample = features[0]["labels"]

            if not isinstance(sample, list):  # one task
                batch["labels"] = torch.tensor([i["labels"] for i in features])
            elif len(sample) == 1:
                batch["labels"] = torch.from_numpy(
                    np.concatenate([i["labels"] for i in features])
                )
            else:  # multi task classification, left to float to keep the NaNs
                batch["labels"] = torch.from_numpy(
                    np.stack([i["labels"] for i in features])
                )

        return batch
