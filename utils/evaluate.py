from sklearn.metrics import roc_auc_score
import numpy as np
from functools import partial

import tracemalloc
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import wandb

def multi_label_metrics(eval_pred: tuple, label_names):
    """
    Computes ROC AUC and accuracy per label, as well as their averages.
    
    Args:
        eval_pred: Tuple of (logits, labels) from the evaluation dataloader.
        label_names: List of label names.

    Returns:
        metrics: Dictionary of metrics.
    """

    logits, labels = eval_pred
    logits = logits[0]
    probs = 1 / (1 + np.exp(-logits))

    # Create mask for valid (non-NaN) labels
    valid_labels_mask = ~np.isnan(labels)

    roc_auc_per_label = {}
    accuracy_per_label = {}
    for label_idx in range(labels.shape[1]):  # Iterate over each label
        # Use the mask to select only valid labels and corresponding probabilities
        valid_probs = probs[valid_labels_mask[:, label_idx], label_idx]
        valid_labels = labels[valid_labels_mask[:, label_idx], label_idx]

        if len(valid_labels) > 0:  # Only compute metrics if there are valid labels
            roc_auc_per_label[f"{label_names[label_idx]}_roc_auc"] = roc_auc_score(
                valid_labels, valid_probs
            )

            predicted_labels = valid_probs > 0.5
            accuracy_per_label[f"{label_names[label_idx]}_accuracy"] = (
                predicted_labels == valid_labels
            ).mean()

    # Organize metrics into a nested dictionary
    
    mean_metrics = {
        "mean_roc_auc": np.mean(list(roc_auc_per_label.values())),
        "mean_accuracy": np.mean(list(accuracy_per_label.values())),
    }

    return {**mean_metrics, **roc_auc_per_label, **accuracy_per_label}

    def regression_metrics(eval_pred, true_value):
        pass


def prepare_evaluation_for_training(pretraining: bool, dataset_name: str, **kwargs):
    if not pretraining:
        if dataset_name in ["tox21_original", "tox21"]:
            return partial(
                multi_label_metrics,
                label_names=[
                    "NR.AhR",
                    "NR.AR",
                    "NR.AR.LBD",
                    "NR.Aromatase",
                    "NR.ER",
                    "NR.ER.LBD",
                    "NR.PPAR.gamma",
                    "SR.ARE",
                    "SR.ATAD5",
                    "SR.HSE",
                    "SR.MMP",
                    "SR.p53",
                ],
            )
        if dataset_name == "ZINC":
            label_name = "penalized logP"
            return None
        if dataset_name == "qm9":
            label_names = [
                "A",
                "B",
                "C",
                "mu",
                "alpha",
                "homo",
                "lumo",
                "gap",
                "r2",
                "zpve",
                "u0",
                "u298",
                "h298",
                "g298",
                "cv",
                "u0_atom",
                "u298_atom",
                "h298_atom",
                "g298_atom",
            ]
            return None
        raise ValueError("Invalid dataset name for fine tuning.")
    else:
        if dataset_name == "pcqm4mv2":
            return None
        if dataset_name == "pcba":
            return None
        if dataset_name == "qm9":
            return None
        raise ValueError("Invalid dataset name for pretraining.")


# Define your custom callback
class MemoryProfilerCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Check the current training step
        if state.global_step % 500 == 0:
            # Start tracing memory allocations
            tracemalloc.start()

            # Get snapshot of current memory consumption
            snapshot = tracemalloc.take_snapshot()

            # Display the top 10 lines consuming the memory
            top_stats = snapshot.statistics('lineno')

            for i, stat in enumerate(top_stats[:30]):
                wandb.log({f'memory_stat_{i}': str(stat)}, step=state.global_step)
                print(stat)

            # Stop tracing memory allocations
            tracemalloc.stop()