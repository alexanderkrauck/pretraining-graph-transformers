# TODO: consider adding the download of the data to this script
# TODO: consider adding the huggingface data preprocessing to this script
import os
from utils import data as data_utils
import torch_geometric as pyg
from utils import ZincWithRDKit
from datasets import DatasetDict, load_from_disk
from os.path import join
import subprocess
from utils import graphormer_data_collator_improved as graphormer_collator_utils
from argparse import ArgumentParser


def load_all_data(data_dir: str = "data", create_conformers: bool = True):
    print(f"loading all data to {data_dir}. This may take a while.")

    # load pcqm4mv2 data
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
            join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf"),
        ]
    )

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
    print("downloading zinc data")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="train")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="val")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="test")
    del zinc

    # Here start the data processing to arrow format
    print("processing pcqm4mv2 data to arrow format")
    ds = data_utils.sdf_to_arrow(
        join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf"),
        to_disk_location=join(data_dir, "pcqm4mv2/processed/arrow"),
    )

    print("processing tox21 original data to arrow format")
    ds_train = data_utils.sdf_to_arrow(
        join(data_dir, "tox21_original/raw/tox21.sdf"),
        to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
        target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
        split_column=4,
        take_split="training",
    )
    ds_val = data_utils.sdf_to_arrow(
        join(data_dir, "tox21_original/raw/tox21.sdf"),
        to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
        target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
        split_column=4,
        take_split="validation",
    )
    ds_test = data_utils.sdf_to_arrow(
        join(data_dir, "tox21_original/raw/tox21.sdf"),
        to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
        target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
        split_column=4,
        take_split="test",
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
    ds = data_utils.csv_to_arrow(
        join(data_dir, "tox21/raw/tox21.csv"),
        to_disk_location=join(data_dir, "tox21/processed/arrow"),
        include_conformer=create_conformers,
        id_column=12,
    )

    print("processing pcba data to arrow format")
    ds = data_utils.csv_to_arrow(
        join(data_dir, "pcba/raw/pcba.csv"),
        include_conformer=create_conformers,  # NOTE: for now because its very slow
        to_disk_location=join(data_dir, "pcba/processed/arrow"),
        smiles_column=-1,
        id_column=-2,
        target_columns=list(range(0, 128)),
    )

    print("processing qm9 data to arrow format")
    ds = data_utils.sdf_to_arrow(
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
    )

    print("processing zinc data to arrow format")
    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="train")
    mol_list = zinc.to_rdkit_molecule_list()
    ds_train = data_utils.rdkit_to_arrow(
        mol_list,
        target_list=zinc.y.numpy().tolist(),
    )

    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="val")
    mol_list = zinc.to_rdkit_molecule_list()
    ds_val = data_utils.rdkit_to_arrow(
        mol_list,
        target_list=zinc.y.numpy().tolist(),
    )

    zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="test")
    mol_list = zinc.to_rdkit_molecule_list()
    ds_test = data_utils.rdkit_to_arrow(
        mol_list,
        target_list=zinc.y.numpy().tolist(),
    )

    dataset_dict = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_val,
            "test": ds_test,
        }
    )

    dataset_dict.save_to_disk(join(data_dir, "ZINC/processed/arrow"))

    # Here start the mapping of the data to input format to the model
    print("mapping data to input format to the model")

    print("mapping pcqm4mv2 data to input format to the model")
    data_utils.map_arrow_dataset_from_disk(
        join(data_dir, "pcqm4mv2/processed"), is_dataset_dict=False
    )

    print("mapping tox21_original data to input format to the model")
    data_utils.map_arrow_dataset_from_disk(
        join(data_dir, "tox21_original/processed"), is_dataset_dict=True
    )

    print("mapping tox21 data to input format to the model")
    data_utils.map_arrow_dataset_from_disk(
        join(data_dir, "tox21/processed"), is_dataset_dict=False
    )

    print("mapping pcba data to input format to the model")
    data_utils.map_arrow_dataset_from_disk(
        join(data_dir, "qm9/processed"), is_dataset_dict=False
    )

    print("mapping qm9 data to input format to the model")
    data_utils.map_arrow_dataset_from_disk(
        join(data_dir, "ZINC/processed"), is_dataset_dict=True
    )

    print("mapping pcba data to input format to the model")
    data_utils.map_arrow_dataset_from_disk(
        join(data_dir, "pcba/processed"), is_dataset_dict=False
    )

    print("\nData loading finished!\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        help="Directory where all data should be stored.",
        default=f"data",
        type=str,
    )

    parser.add_argument(
        "-c",
        "--conformers",
        help="whether or not conformers will be created if no 3d conig is here yet.",
        default=False,
        type=bool,
    )

    args = parser.parse_args()
    load_all_data(**dict(args._get_kwargs()))