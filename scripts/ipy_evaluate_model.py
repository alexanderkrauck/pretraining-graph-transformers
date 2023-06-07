#%%
# Standard library imports
import os
os.chdir("..")
import random
from argparse import ArgumentParser

# Third party imports
import numpy as np
import torch
import yaml
from transformers import (
    GraphormerConfig,
    GraphormerForGraphClassification,
    Trainer,
    TrainingArguments,
)

# Local application imports
from utils import data as data_utils
from utils import graphormer_data_collator_improved as graphormer_collator_utils
from utils import setup as setup_utils
from utils import evaluate as evaluate_utils

#%%
data_dir = "/system/user/publicwork/student/krauck/graph_data/"
model = GraphormerForGraphClassification.from_pretrained("/system/user/publicwork/student/krauck/graph_data/runs/07-06-2023_12-27-56_zinc_no_pretrain/checkpoints/checkpoint-18000")

# %%
dataset = data_utils.prepare_dataset_for_training(
        False, dataset_name = "ZINC", data_dir=data_dir)
evaluation_func = evaluate_utils.prepare_evaluation_for_training(
        False, dataset_name = "ZINC"
    )

training_args = TrainingArguments(
    output_dir='./logs',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',
    report_to = []            # directory for storing logs
)

# Specify the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    compute_metrics=evaluation_func,
    data_collator=graphormer_collator_utils.GraphormerDataCollator(num_edge_features=3),
               # evaluation dataset
)



# Evaluate the model

# %%
eval_results = trainer.evaluate(eval_dataset=dataset["test"])
eval_results
# %%
