#!/usr/bin/python
"""This script provides the ability to download and preprocess all datasets used in the project so experiments can be run.

Copyright (c) 2023 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"

import os
from utils import preprocessing as preprocessing_utils
from utils import ZincWithRDKit
from datasets import DatasetDict
from os.path import join
import subprocess
from argparse import ArgumentParser


def load_all_data(data_dir: str = "data", create_conformers: bool = True, process_to_model_input: bool = False):
    print(f"loading all data to {data_dir}. This may take a while.")

    #TODO: consider adding checks for the data already being downloaded.
    #Low priority tho, as this is not a focus of the project.

    # load pcqm4mv2 data

    if not os.path.isfile(join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf")):
        print("downloading pcqm4mv2 data")
        subprocess.run(
            [
                "wget",
                "-P",
                join(data_dir, "pcqm4mv2/raw"),
                "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz",
            ]
        )
        subprocess.run(
            [
                "tar",
                "-xf",
                join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf.tar.gz"),
                "-C",
                join(data_dir, "pcqm4mv2/raw/"),
            ]
        )

    if not os.path.isfile(join(data_dir, "tox21_original/raw/tox21.sdf")):
        # load tox21 original data
        print("downloading tox21 original data")
        subprocess.run(
            [
                "wget",
                "-P",
                join(data_dir, "tox21_original/raw"),
                "http://bioinf.jku.at/research/DeepTox/tox21_compoundData.csv",
            ]
        )
        subprocess.run(
            [
                "wget",
                "-P",
                join(data_dir, "tox21_original/raw"),
                "http://bioinf.jku.at/research/DeepTox/tox21.sdf.gz",
            ]
        )
        subprocess.run(["gunzip", join(data_dir, "tox21_original/raw/tox21.sdf.gz")])

    # load tox21 data
    if not os.path.isfile(join(data_dir, "tox21/raw/tox21.csv")):
        print("downloading tox21 data")
        subprocess.run(
            [
                "wget",
                "-P",
                join(data_dir, "tox21/raw"),
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
            ]
        )
        subprocess.run(["gunzip", join(data_dir, "tox21/raw/tox21.csv.gz")])

    # load pcba data
    if not os.path.isfile(join(data_dir, "pcba/raw/pcba.csv")):
        print("downloading pcba data")
        subprocess.run(
            [
                "wget",
                "-P",
                join(data_dir, "pcba/raw"),
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
            ]
        )
        subprocess.run(["gunzip", join(data_dir, "pcba/raw/pcba.csv.gz")])

    # load qm9 data
    if not os.path.isfile(join(data_dir, "qm9/raw/QM9_README")):
        print("downloading qm9 data")
        subprocess.run(
            [
                "wget",
                "-P",
                join(data_dir, "qm9/raw"),
                "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
            ]
        )
        subprocess.run(
            ["unzip", join(data_dir, "qm9/raw/qm9.zip"), "-d", join(data_dir, "qm9/raw")]
        )
        subprocess.run(["rm", join(data_dir, "qm9/raw/qm9.zip")])

    # load zinc data
    print("downloading zinc data (if not downloaded already)")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="train")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="val")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="test")
    del zinc

    # Here start the data processing to arrow format
    print("processing pcqm4mv2 data to arrow format")
    ds = preprocessing_utils.sdf_to_arrow(
        join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf"),
        to_disk_location=join(data_dir, "pcqm4mv2/processed/arrow"),
        cache_dir=join(data_dir, "huggingface"),
    )

    print("processing tox21 original data to arrow format")
    ds_train = preprocessing_utils.sdf_to_arrow(
        join(data_dir, "tox21_original/raw/tox21.sdf"),
        to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
        target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
        split_column=4,
        take_split="training",
        cache_dir=join(data_dir, "huggingface")
    )
    ds_val = preprocessing_utils.sdf_to_arrow(
        join(data_dir, "tox21_original/raw/tox21.sdf"),
        to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
        target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
        split_column=4,
        take_split="validation",
        cache_dir=join(data_dir, "huggingface")
    )
    ds_test = preprocessing_utils.sdf_to_arrow(
        join(data_dir, "tox21_original/raw/tox21.sdf"),
        to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
        target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
        split_column=4,
        take_split="test",
        cache_dir=join(data_dir, "huggingface")
    )

    dataset_dict = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )
    dataset_dict.save_to_disk(join(data_dir, "tox21_original/processed/arrow"))

    print("processing tox21 data to arrow format")
    ds = preprocessing_utils.csv_to_arrow(
        join(data_dir, "tox21/raw/tox21.csv"),
        to_disk_location=join(data_dir, "tox21/processed/arrow"),
        include_conformer=create_conformers,
        id_column=12,
        target_columns = [0,1,2,3,4,5,6,7,8,9,10,11],
        cache_dir=join(data_dir, "huggingface")
    )

    print("processing pcba data to arrow format")
    ds = preprocessing_utils.csv_to_arrow(
        join(data_dir, "pcba/raw/pcba.csv"),
        include_conformer=create_conformers,  # NOTE: for now because its very slow
        to_disk_location=join(data_dir, "pcba/processed/arrow"),
        smiles_column=-1,
        id_column=-2,
        target_columns=list(range(0, 128)),
        cache_dir=join(data_dir, "huggingface")
    )

    print("processing qm9 data to arrow format")
    ds = preprocessing_utils.sdf_to_arrow(
        join(data_dir, "qm9/raw/gdb9.sdf"),
        to_disk_location=join(data_dir, "qm9/processed/arrow"),
        csv_with_metainfo=join(data_dir, "qm9/raw/gdb9.sdf.csv"),
        target_columns=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
        cache_dir=join(data_dir, "huggingface")
    )

    print("processing zinc data to arrow format")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="train")
    mol_list = zinc.to_rdkit_molecule_list()
    ds_train = preprocessing_utils.rdkit_to_arrow(
        mol_list,
        target_list=zinc.y.numpy().tolist(),
        cache_dir=join(data_dir, "huggingface")
    )

    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="val")
    mol_list = zinc.to_rdkit_molecule_list()
    ds_val = preprocessing_utils.rdkit_to_arrow(
        mol_list,
        target_list=zinc.y.numpy().tolist(),
        cache_dir=join(data_dir, "huggingface")
    )

    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="test")
    mol_list = zinc.to_rdkit_molecule_list()
    ds_test = preprocessing_utils.rdkit_to_arrow(
        mol_list,
        target_list=zinc.y.numpy().tolist(),
        cache_dir=join(data_dir, "huggingface")
    )

    dataset_dict = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )

    dataset_dict.save_to_disk(join(data_dir, "ZINC/processed/arrow"))

    if process_to_model_input:
        # Here start the mapping of the data to input format to the model
        print("mapping data to input format to the model")

        print("mapping pcqm4mv2 data to input format to the model")
        preprocessing_utils.map_arrow_dataset_from_disk(
            join(data_dir, "pcqm4mv2/processed"), is_dataset_dict=False
        )

        print("mapping tox21_original data to input format to the model")
        preprocessing_utils.map_arrow_dataset_from_disk(
            join(data_dir, "tox21_original/processed"), is_dataset_dict=True
        )

        print("mapping tox21 data to input format to the model")
        preprocessing_utils.map_arrow_dataset_from_disk(
            join(data_dir, "tox21/processed"), is_dataset_dict=False
        )

        print("mapping pcba data to input format to the model")
        preprocessing_utils.map_arrow_dataset_from_disk(
            join(data_dir, "qm9/processed"), is_dataset_dict=False
        )

        print("mapping qm9 data to input format to the model")
        preprocessing_utils.map_arrow_dataset_from_disk(
            join(data_dir, "ZINC/processed"), is_dataset_dict=True
        )

        print("mapping pcba data to input format to the model")
        preprocessing_utils.map_arrow_dataset_from_disk(
            join(data_dir, "pcba/processed"), is_dataset_dict=False
        )

    print("\nData loading finished!\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        help="Directory where all data should be stored.",
        default=f"data",
        type=str,
    )

    parser.add_argument(
        "-c",
        "--create_conformers",
        help="whether or not conformers will be created if no 3d conig is here yet.",
        default=False,
        type=bool,
    )

    args = parser.parse_args()
    load_all_data(**dict(args._get_kwargs()))
