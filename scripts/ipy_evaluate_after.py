#%%
import os
os.chdir("..")
import random
from argparse import ArgumentParser

# Third party imports
import numpy as np
import torch
import yaml
from utils.modeling_graphormer_improved import (
    GraphormerConfig,
    GraphormerForGraphClassification,
    BetterGraphormerConfig
)

from transformers import (    Trainer,
    TrainingArguments,)

from utils.modeling_graphormer_improved_3d import Graphormer3DForGraphClassification, Graphormer3DConfig
from utils.graphormer_data_collator_improved_3d import Graphormer3DDataCollator   


# Local application imports
from utils import data as data_utils
from utils import graphormer_data_collator_improved as graphormer_collator_utils
from utils import setup as setup_utils
from utils import evaluate as evaluate_utils
#%%
import wandb
run = wandb.init()
data_dir = "data/"
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
#%%
def do_evaluation(art_names, evaluation_func, target_scaler = None, model_type="graphormer"):
    eval_results_list = []
    for artifact_name in art_names:
        artifact = run.use_artifact(artifact_name, type='model')
        artifact_dir = artifact.download()

        if model_type == "graphormer":
            config = BetterGraphormerConfig.from_pretrained(artifact_dir)
        else:
            config = Graphormer3DConfig.from_pretrained(artifact_dir)  
        config.classification_task = "regression"
        model_params = torch.load(artifact_dir + "/pytorch_model.bin", torch.device('cpu'))
        if model_type == "graphormer":
            model = GraphormerForGraphClassification(config)
        else:
            model = Graphormer3DForGraphClassification(config)
        model.load_state_dict(model_params)

        if model_type == "graphormer":
            collator = graphormer_collator_utils.GraphormerDataCollator(model_config=model.config, on_the_fly_processing=False, collator_mode="classification", target_scaler=target_scaler)
        else:
            collator = Graphormer3DDataCollator(model_config=model.config, on_the_fly_processing=False, collator_mode="classification", target_scaler=target_scaler)

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            compute_metrics=evaluation_func,
            data_collator=collator,
                    # evaluation dataset
        )
        eval_results = trainer.evaluate(eval_dataset=dataset["test"])
        eval_results_list.append(eval_results)

    for key in eval_results_list[0].keys():
        print(key)
        listed_results = [eval_result[key] for eval_result in eval_results_list]
        print(listed_results)
        print(np.mean(listed_results), np.std(listed_results))
# %% ZINC
dataset = data_utils.prepare_dataset_for_training(False, seed=42, memory_mode="full", dataset_name = "ZINC", data_dir=data_dir, model_type="graphormer")
evaluation_func = evaluate_utils.prepare_evaluation_for_training(
        False, dataset_name = "ZINC"
    )


# %%
artifact_names = ["alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v7",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v6",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v5",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v4",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v3",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v2",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v1",
                  "alexanderkrauck/pretrained_graph_transformer/model-04-07-2023_10-29-34_51_zinc_on_pretrained:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-03-07-2023_07-47-10_zinc_finetuning_on_pcqm4mv2:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-02-07-2023_14-23-13_zinc_finetuning_on_pcqm4mv2:v0"]

#%%
do_evaluation(artifact_names, evaluation_func, None, model_type="graphormer")

# %%
artifact_names = ["alexanderkrauck/pretrained_graph_transformer/model-09-07-2023_15-44-48_69_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-11-07-2023_18-12-36_70_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-14-07-2023_14-43-01_72_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-15-07-2023_05-10-14_73_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-15-07-2023_14-40-51_74_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-16-07-2023_05-10-10_75_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-16-07-2023_19-56-28_76_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-17-07-2023_05-46-54_77_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-18-07-2023_08-26-14_99_zinc_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-18-07-2023_17-45-03_100_zinc_no_pretrain:v0"]



#%% ZINC 3D
dataset = data_utils.prepare_dataset_for_training(False, seed=42, memory_mode="full", dataset_name = "ZINC", data_dir=data_dir, model_type="graphormer3d")
evaluation_func = evaluate_utils.prepare_evaluation_for_training(
        False, dataset_name = "ZINC"
    )
#%% no pretrain
artifact_names = ["alexanderkrauck/pretrained_graph_transformer/model-19-07-2023_19-32-24_124_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-19-07-2023_08-39-43_123_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-15-07-2023_23-22-50_62_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-15-07-2023_08-09-58_61_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-14-07-2023_14-18-05_60_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-13-07-2023_17-30-33_59_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-13-07-2023_07-11-44_58_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-12-07-2023_19-33-46_57_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-12-07-2023_08-12-09_56_zinc_run_3d_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-11-07-2023_18-11-42_55_zinc_run_3d_no_pretrain:v0"]

do_evaluation(artifact_names, evaluation_func, None, model_type = "graphormer3d")

#%% pretrained on pcqm4mv2
artifact_names = ["alexanderkrauck/pretrained_graph_transformer/model-28-07-2023_20-23-30_72_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-28-07-2023_12-34-18_71_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-28-07-2023_06-28-05_70_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-24-07-2023_11-47-23_57_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-24-07-2023_08-31-35_56_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-24-07-2023_02-57-46_55_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-23-07-2023_21-52-28_54_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-23-07-2023_14-47-07_53_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-23-07-2023_11-39-22_52_ZINC_finetune_3d_noise_as_paper:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-23-07-2023_07-10-21_51_ZINC_finetune_3d_noise_as_paper:v0"]

do_evaluation(artifact_names, evaluation_func, None, model_type = "graphormer3d")

# %% QM9

dataset = data_utils.prepare_cv_dataset_for_training(seed=72, memory_mode="full", dataset_name = "qm9", data_dir=data_dir, model_type="graphormer", num_folds=10, train_split=0.9)[0]
target_scaler = data_utils.get_regression_target_scaler(dataset["train"], num_classes=19)
evaluation_func = evaluate_utils.prepare_evaluation_for_training(
        False, dataset_name = "qm9", target_scaler=target_scaler
    )

# %%
artifact_names = ["alexanderkrauck/pretrained_graph_transformer/model-21-07-2023_22-24-14_81_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-21-07-2023_16-20-19_80_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-21-07-2023_01-29-44_79_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-20-07-2023_10-54-22_78_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-19-07-2023_22-29-30_77_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-19-07-2023_16-23-36_76_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-18-07-2023_21-19-58_75_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-18-07-2023_09-32-49_74_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-17-07-2023_14-45-03_73_qm9_no_pretrain:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-17-07-2023_07-44-21_72_qm9_no_pretrain:v0"]
# %%
do_evaluation(artifact_names, target_scaler)
# %%
dataset = data_utils.prepare_cv_dataset_for_training(seed=72, memory_mode="full", dataset_name = "qm9", data_dir=data_dir, model_type="graphormer3d", num_folds=10, train_split=0.9)[0]
target_scaler = data_utils.get_regression_target_scaler(dataset["train"], num_classes=19)
evaluation_func = evaluate_utils.prepare_evaluation_for_training(
        False, dataset_name = "qm9", target_scaler=target_scaler
    )

artifact_names = ["alexanderkrauck/pretrained_graph_transformer/model-19-07-2023_09-54-17_81_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-19-07-2023_03-20-20_80_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-18-07-2023_13-12-46_79_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-17-07-2023_23-04-39_78_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-17-07-2023_08-56-48_77_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-17-07-2023_03-08-54_76_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-16-07-2023_19-58-48_75_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-16-07-2023_17-02-10_74_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-16-07-2023_10-32-26_73_qm9_no_pretrain_3d:v0",
                  "alexanderkrauck/pretrained_graph_transformer/model-16-07-2023_08-42-30_72_qm9_no_pretrain_3d:v0"]
# %%
# %%
do_evaluation(artifact_names, target_scaler, model_type = "graphormer3d")
# %%
