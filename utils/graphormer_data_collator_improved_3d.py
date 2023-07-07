# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

from typing import Any, Dict, List, Mapping

import numpy as np
import torch


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


def preprocess_3D_item(
    item, single_embedding_offset: int = 512, **kwargs
):
    """
    Preprocess a single item from a dataset.

    Args:
        item (Dict[str, Any]): A single item from a dataset.
        num_edge_features (int): The number of edge features to use. NOTE: This is mainly here for the speical case of graphs without any edges.
        single_embedding_offset (int): The offset to use for the single embedding.
    """

    # Transfer the data from the item to numpy arrays.
    # below assumes that the input has edge_attributes and node_features
    node_feature = np.asarray(item["node_feat"], dtype=np.int64)

    # Create node features with an offset.
    input_nodes = convert_to_single_emb(node_feature, single_embedding_offset) + 1



    # Create Dictionary entries with all the data.
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding

    pos = np.asarray(item["pos"], dtype= np.float64)
    item["pos"] = pos

    # This does not make much sense to me, but I, for now, don't want to rewrite the model.
    if "target" in item:
        item["labels"] = item["target"]

    return item


class Graphormer3DDataCollator:
    def __init__(
        self,
        model_config,
        on_the_fly_processing: bool = True,
        collator_mode: str = "classifcation",
    ):
        """
        Data collator for Graphormer.

        Args:
        -----
        on_the_fly_processing : bool
            If true, the preprocessing is done on the fly. If false, the data is expected to be processed already.
            If True, the TrainingArguments need to have the following parameters set: remove_unused_columns = False, num_edge_features = num_edge_features
        collator_mode : str
            The collator mode to use. Can be either "classification", "inference" or "pretraining".
        """


        self.on_the_fly_processing = on_the_fly_processing
        self.collator_mode = collator_mode
        self.mask_prob = model_config.mask_prob
        self.single_embedding_offset = model_config.single_embedding_offset

    def __call__(self, features: List[dict]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]

        if self.on_the_fly_processing:
            features = [
                preprocess_3D_item(f, self.single_embedding_offset)
                for f in features
            ]

        batch = {}

        # NOTE: per preprocessing edge_feat_size and edge_input_size is always the same.
        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        # TODO maybe remove this if check
         
        batch_size = len(features)


        # Create tensor for edge attribute (type) data

        # Create tensor for spatial position data (shortest path distance)
 
        # Create tensor for node feature data
        batch["input_nodes"] = torch.zeros(
            batch_size, max_node_num, node_feat_size, dtype=torch.long
        )
        batch["pos"] = torch.zeros(batch_size, max_node_num, 3, dtype=torch.float64)
        # NOTE: Here again input edges which is huge. not sure yet why.


        n_node_list = []

        # if self.collator_mode == "pretraining":
        #     # NOTE: this is only for pretraining
        #     batch["mask"] = torch.zeros(
        #         batch_size, max_node_num, dtype=torch.long
        #     ).bool()

        for ix, f in enumerate(features):


            f_input_nodes = torch.from_numpy(f["input_nodes"])
            f_pos = torch.from_numpy(f["pos"])


            n_nodes = f_input_nodes.shape[0]
            n_node_list.append(n_nodes)

                # so all that are above the max are set to -inf #TODO: f["attn_bias"] is completely useless. you can just use the spatial pos
            # TODO: this is all square matrices so we could just use one dim

            batch["input_nodes"][ix, :n_nodes] = f_input_nodes
            batch["pos"][ix, :n_nodes] = f_pos
            # TODO: this if check sorts out graphs without any edges that are in bad format. Maybe remove later.
            

            # if self.collator_mode == "pretraining":
            #     mask = torch.rand(n_nodes) < self.mask_prob
            #     if not torch.any(mask):
            #         mask[torch.randint(0, n_nodes, (1,))] = True
            #     batch["mask"][ix, :n_nodes] = mask

        batch["n_nodes"] = torch.tensor(n_node_list, dtype=torch.long)

        # Only add labels if they are in the features. For inference, or pretraining, the features won't have labels.
        # if self.collator_mode == "pretraining":

        #     batch["labels"] = batch["input_nodes"][batch["mask"]]
        #     for i in range(batch["labels"].shape[1]):
        #         batch["labels"][:, i] = (
        #             batch["labels"][:, i] - i * self.single_embedding_offset
        #         )

        #     batch["input_nodes"][batch["mask"]] = self.single_embedding_offset - 1
        #     batch["n_masked_nodes"] = torch.sum(batch["mask"], dim=-1)

        if self.collator_mode == "classification":
            sample = features[0]["labels"]

            if not isinstance(sample, (list, np.ndarray)):  # one task
                batch["labels"] = torch.tensor([i["labels"] for i in features])
            elif len(sample) == 1:
                batch["labels"] = torch.from_numpy(
                    np.concatenate([i["labels"] for i in features])
                )
            else:  # multi task classification, left to float to keep the NaNs
                batch["labels"] = torch.from_numpy(
                    np.stack([i["labels"] for i in features])
                )

        if not self.collator_mode == "inference":
            batch["return_loss"] = True

        return batch
