#!/usr/bin/python
"""Main module for the project."""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"



import utils
import yaml
import os
import shutil


from datetime import datetime
from argparse import ArgumentParser

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
    print(f"YAML file: {yaml_file}")

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

    print(f"Experiment {name} started.")
    print(f"Experiment Log directory: {logpath}")



    

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
