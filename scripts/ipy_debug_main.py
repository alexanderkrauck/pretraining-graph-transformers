#%%
import os
os.chdir("..")

from main import main

# %%
main(name = "*time*_test", logdir = "runs", yaml_file="configs/dummy_config.yml")
# %%
