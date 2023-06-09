#%%
import os
os.chdir("..")

from main import main

#%%
name = "*time*_test"
logdir = "runs"
yaml_file="configs/dummy_config.yml"
from_pretrained = None

# %%
main(name = name, logdir = logdir, yaml_file=yaml_file)
# %%Here if i want to debug the main.py script in ipython
# Standard library imports
import os
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
import wandb

# Local application imports
from utils import data as data_utils
from utils import graphormer_data_collator_improved as graphormer_collator_utils
from utils import setup as setup_utils
from utils import evaluate as evaluate_utils
#%%
with open(yaml_file, "r") as file:
    config = yaml.safe_load(file)

name = setup_utils.get_experiment_name(config, name)
logpath = os.path.join(logdir, name)
logger = setup_utils.setup_logging(logpath, name, yaml_file)
setup_utils.setup_wandb(name, logdir)

seed = config["seed"]  # TODO: maybe allow multiple seeds with a loop
logger.info(f"Set the random seed to : {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# TODO: move below to a seperate function maybe
pretraining = config["pretraining"]
dataset = data_utils.prepare_dataset_for_training(
    pretraining, **config["data_args"]
)
evaluation_func = evaluate_utils.prepare_evaluation_for_training(
    pretraining, **config["data_args"]
)
#%%
# Set up the training arguments
# TODO: maybe add logic for hyperparameter search

device_count = torch.cuda.device_count()
logger.info(f"Using CUDA devices: {[torch.cuda.get_device_properties(device_index) for device_index in range(device_count)]}")
setup_utils.setup_batch_size(config, device_count)


training_args = TrainingArguments(
    output_dir=os.path.join(logpath, "checkpoints"),
    logging_dir=logpath,
    seed=seed,
    data_seed=seed,
    run_name=name,
    report_to=["wandb", "tensorboard"],
    **config["trainer_args"],
)

if "labels" in dataset["train"].column_names:
    label_0 = dataset["train"][0]["labels"]
else:
    label_0 = dataset["train"][0]["target"]
if not isinstance(label_0, list):
    n_classes = 1
else:
    n_classes = len(label_0)
#%%
# Define your custom model here
if from_pretrained is not None:
    model = GraphormerForGraphClassification.from_pretrained(
        from_pretrained, num_classes=n_classes, ignore_mismatched_sizes=True
    )
else:
    model_config = GraphormerConfig(
        num_classes=n_classes, **config["model_args"]
    )
    model = GraphormerForGraphClassification(model_config)

setup_utils.log_model_params(model, logger)
on_the_fly_processing = False if config["data_args"]["memory_mode"]=="full" else True
collator = graphormer_collator_utils.GraphormerDataCollator(num_edge_features=3, on_the_fly_processing=on_the_fly_processing)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
    compute_metrics=evaluation_func,
)
#%%
trainer.train()
#%%
# Do a test run #TODO: maybe put it in a seperate function/file
test_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        report_to=[],
        output_dir=os.path.join(logpath, "checkpoints"),
        **config["trainer_args"],
    ),
    data_collator=collator,
    compute_metrics=evaluation_func,
)
test_results = trainer.evaluate(dataset["test"])

# Prefix all keys in the dictionary with 'test_'
test_results = {f"test_{k}": v for k, v in test_results.items()}

# Log the results
wandb.log(test_results)