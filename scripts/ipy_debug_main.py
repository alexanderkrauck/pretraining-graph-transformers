#%%
import os
os.chdir("..")

from main import main
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import utils.data as data_utils

#%%
name = "*time*_test_pretrain"
logdir = "runs"
yaml_file="configs/zinc_run_without_pretrain_3d.yml"
from_pretrained = "/home/alexander/server_runs/19-07-2023_19-32-24_124_zinc_run_3d_no_pretrain/checkpoints/checkpoint-165500"
model_type = "graphormer3d"

# %%
trainer = main(name = name, logdir = logdir, yaml_file=yaml_file, model_type=model_type, return_trainer_instead=True, from_pretrained=from_pretrained)
#%%
dataset = data_utils.prepare_dataset_for_training(False, seed=42, memory_mode="full", dataset_name = "ZINC", data_dir="data/", model_type="graphormer3d")
#%%
trainer.evaluate(dataset["test"])

#%%
dl = trainer.get_train_dataloader()
#%%
def test_dl_speed(data_loader, times= -1, append_batches = False):
    i = 0
    _start = time.time()
    batches = []
    for batch in tqdm(data_loader):
        _end = time.time()
        if append_batches:
            batches.append(batch)
        #print(f"Time taken for batch {i}: {_end-_start}")

        _start = time.time()

        i+=1
        if i==times:
            break
    return batches

#%%
test_dl_speed(dl)
#%%
dataset = trainer.train_dataset
dataset = dataset.shuffle()

dl2 = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        #sampler=None,
        collate_fn=trainer.data_collator,
        drop_last=False,
        #num_workers=1,
        #pin_memory=True,
        #worker_init_fn=dl.worker_init_fn,
    )
test_dl_speed(dl2, times=10)
#%%
dataset = trainer.train_dataset
dataset = dataset.shuffle()
tot_batches = 10
batch_size = 256
dataset_size = len(dataset)
batches2 = []
for e in tqdm(range(tot_batches)):
    data_batch = [dataset[(i + e * batch_size) % dataset_size] for i in range(batch_size)]
    #data_batch = trainer.data_collator(data_batch)
    batches2.append(data_batch)

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

wandb.config.update(config["data_args"])
wandb.config.pretraining = pretraining

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


#%%
import wandb

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
os.environ["WANDB_LOG_MODEL"]="true"



wandb.init(project = "test", name="test21", dir=os.path.join("test_stuff", "wandb"))

# %%
