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


# NOTE:  Not really sure about this function or the purpose of it.
#       It basically adds multiples of the offset to the features of each sample.
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def preprocess_item(item, keep_features=True):
    requires_backends(preprocess_item, ["cython"])

    # Transfer the data from the item to numpy arrays.
    # below assumes that the input has edge_attributes and node_features
    edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    num_nodes = item["num_nodes"]

    # Create node features with an offset.
    input_nodes = convert_to_single_emb(node_feature) + 1

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]

    # Create dense edge_attr matrix with an offset.
    attn_edge_type = np.zeros(
        [num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64
    )
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
    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    attn_bias = np.zeros(
        [num_nodes + 1, num_nodes + 1], dtype=np.single
    )  # with graph token

    # Create Dictionary entries with all the data.
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
    item["attn_bias"] = attn_bias
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
    def __init__(self, spatial_pos_max=20):
        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        self.spatial_pos_max = spatial_pos_max

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}

        # NOTE: per preprocessing edge_feat_size and edge_input_size is always the same.
        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        edge_feat_size = len(features[0]["attn_edge_type"][0][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in features)
        edge_input_size = len(features[0]["input_edges"][0][0][0])
        batch_size = len(features)

        # Here things are scaled to the maximum size of the batch as I did before with OHG.
        # This kind of makes parts of the preprocessing useless.
        # Set attention bias to zero for all nodes scaled to maximum size + 1.
        batch["attn_bias"] = torch.zeros(
            batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float
        )

        # Create tensor for edge attribute (type) data
        batch["attn_edge_type"] = torch.zeros(
            batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long
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
            edge_input_size,
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
            batch["input_edges"][ix, :n_nodes, :n_nodes, :max_dist] = f["input_edges"]

        batch["out_degree"] = batch["in_degree"]  # NOTE: for undirected graph only

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                batch["labels"] = torch.from_numpy(
                    np.concatenate([i["labels"] for i in features])
                )
            else:  # binary classification
                batch["labels"] = torch.from_numpy(
                    np.concatenate([i["labels"] for i in features])
                )
        else:  # multi task classification, left to float to keep the NaNs
            batch["labels"] = torch.from_numpy(
                np.stack([i["labels"] for i in features])
            )

        return batch
