#!/usr/bin/python
"""Main module for the project."""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"



import utils
import yaml
import os
import shutil

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import GraphormerForGraphClassification, GraphormerConfig


def main(name: str = None, logdir: str = "runs", yaml_file:str = "configs/dummy_config.yml"):
    """
    Main function of the project.

    Args
    ----
        name:(str)
            Name of the experiment.
        logdir (str): Directories where logs are stored.
        yaml_file (str): The yaml file with the config.
    """
    # TODO: Implement the logic of the main function
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    
    if "name" in config:
        if name is None:
            name = config["name"]
        else:
            print(f"Overwriting name {config['name']} with {name} as command line argument is stronger.")
    if name is None:
        name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    logpath = os.path.join(logdir,name)
    if os.path.isdir(logpath):
        timestring = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        name = name + "_" + timestring
        logpath = os.path.join(logdir,name)
        print(f"Experiment {name} already exists, setting name to {name}")
    

    os.makedirs(logpath, exist_ok=False)
    shutil.copy(yaml_file, os.path.join(logpath, "config.yml"))

    current_commit = get_commit() #TODO: Add commit to log in the corresponding experiment folder

    print(f"Experiment {name} started.")
    print(f"Used Config file: {yaml_file}")
    print(f"Experiment Log directory: {logpath}")

    print(f"Current commit: {current_commit}. It is reccomended to only run experiments on a clean commit.")
    print(f"https://github.com/alexanderkrauck/pretrained-graph-transformer/tree/{current_commit}")
    

    #TODO: move below to a seperate function maybe
    dataset = load_dataset('imdb')
    train_dataset = #tokenizer(dataset['train']['text'], truncation=True, padding=True, max_length=512)
    val_dataset = #tokenizer(dataset['test']['text'], truncation=True, padding=True, max_length=512)

    # Define your custom model here
    model_config = GraphormerConfig(
        num_classes = 2,
        embedding_dim = 128,
        num_attention_heads = 8,
        num_hidden_layers = 8
    )

    model = GraphormerForGraphClassification(model_config)

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(logpath,"checkpoints"),          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logpath,            # directory for storing logs
    )


#TODO maybe move to utils or use gitpython
def get_commit(repo_path: str = "."):
    git_folder = Path(repo_path,'.git')
    head_name = Path(git_folder, 'HEAD').read_text().split('\n')[0].split(' ')[-1]
    head_ref = Path(git_folder,head_name)
    commit = head_ref.read_text().replace('\n','')
    return commit

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--name", help="Name of the experiment", default=f"*time*", type=str
    )
    parser.add_argument(
        "-l",
        "--logdir",
        help="Directories where logs are stored",
        default=f"runs",
        type=str,
    )
    parser.add_argument(
        "-y",
        "--yaml_file",
        help="The yaml file with the config",
        default="configs/dummy_config.yml",
        type=str,
    )

    args = parser.parse_args()

    main(**dict(args._get_kwargs()))
